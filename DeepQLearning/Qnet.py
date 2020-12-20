# -*- coding: utf-8 -*-
from abc import ABC

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module, ABC):

    def __init__(self, lr, num_agents, action_size, input_size):
        super(DeepQNetwork, self).__init__()

        """ Set seed for reproducibility """
        T.manual_seed(0)
        self.num_agents = num_agents
        
        """ Shared DNN - Convolutional """
        self.conv1 = nn.Conv2d(4, 16, 3)

        x_test = T.tensor(np.zeros(tuple([1]) + input_size)).float()
        fc_input_size = self.size_of_conv_out(x_test)
        
        """ Shared DNN - Dense """
        
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        
        """ Individual DNN """
            
        self.ff1 = nn.Linear(512, action_size*num_agents)

        if T.cuda.is_available():
            self.device = T.device('cuda')
            print('YOU ARE USING YOUR GPU. LETS HAVE SOME FUN!')
        else:
            self.device = T.device('cpu')
            print('YOUR GPU IS MISSING. POOR CPU. ITS IN CHARGE OF EVERYTHING!')
            
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.SmoothL1Loss()
        self.loss2 = nn.SmoothL1Loss(reduction = 'none')

    def forward(self, x):
        """ Forward function. """
            
        """ Shared DDN - Convolutional """
        x = F.relu(self.conv1(x))
        x = T.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        """ Paralel DDN - Linear """
        Qf = F.relu(self.ff1(x)) # SI NO PONGO UNA CAPA DE ACTIVACIÓN DE SALIDA, EL RESULTADO ES PEOR. MISTERIO MISTERIO! #
        
        return Qf
        

    def size_of_conv_out(self, x):
        """
        Function to extract the output size of the convolutional network.

        :param x: Input of the convolutional network
        :return: Integer with the size of the input of the next layer (FC)
        """

        x = self.conv1(x)
        x = T.flatten(x, start_dim=1)

        return x.shape[1]
