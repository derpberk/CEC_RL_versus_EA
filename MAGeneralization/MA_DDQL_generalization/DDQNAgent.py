# -*- coding: utf-8 -*-
"""

Samuel Yanes Luis. University of Seville.

Class for the *Multi Agent Ypacarai* case and other lacustrine scenarios. PhD research.


"""

import numpy as np
import torch as T
from Qnet import DeepQNetwork
from replay_memory import ReplayBuffer, PrioritizedReplayBuffer
from torch.autograd import Variable


class DDQNAgent(object):

    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, action_joint_dim,
                 mem_size, batch_size, eps_min, eps_dec, replace,
                 prioritized=False, prob_alpha=0.6, beta=0.4, beta_increment=1e-4, 
                 temperature = 0.1, tau = 1e-5):

        """

        Double Deep Q-Learning Agent class.

        -----

        Args:
            gamma: Discount factor for reward. 0 indicates a myopic behaviour. 1 indicates a far-sighted behaviour.
            epsilon: Exploration/exploitation rate. 0 indicates full exploitation.
            lr: Learning Rate. The bigger 'lr' the bigger step in the gradient of the loss.
            n_actions: Number of possible actions.
            input_dims: Dimension of the state (allegedly an image). The channel goes first (CHANN, HEIGHT, WIDTH)
            action_joint_dim: Number of joints for the Multi-agent case. Normally the number of agents.
            mem_size: Number of the Replay Buffer memory.
            batch_size: Number of past experiences used for trainin Q-Network.
            eps_min: Min. value for the exploration.
            eps_dec: Epsilon decay in every epoch.
            replace: Number of epochs for replacing the target network with the behavioral network.

        ------
        """
        
        # Hiperparámetros de entrenamiento #
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta
        self.beta_increment = beta_increment
        self.prob_alpha = prob_alpha
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.update_target_count = replace
        self.action_space = [i for i in range(n_actions)]
        self.action_joint_dim = action_joint_dim
        self.prioritized = prioritized
        self.temperature = temperature
        self.tau = tau

        if not self.prioritized:
            self.memory = ReplayBuffer(mem_size, input_dims, action_joint_dim)
        else:
            self.memory = PrioritizedReplayBuffer(mem_size, input_dims, action_joint_dim, self.prob_alpha)
        
        # Funciones del modelo y del target #
        
        self.q_eval = DeepQNetwork(self.lr, num_agents=action_joint_dim, action_size=n_actions, input_size=input_dims)
        self.q_eval.cuda()
        
        self.q_next = DeepQNetwork(self.lr, num_agents=action_joint_dim, action_size=n_actions, input_size=input_dims)
        self.q_next.cuda()
        
    def store_transition(self, state, action, reward, next_state, done):
        """

        Store a '(s,a,r,s')' transition into the buffer replay.

        ------

        Args:
            state: State of the experience.
            action: Action joint performed in the given 'state'.
            reward: 1D Reward array obtained due to (s,a). One component for every agent.
            next_state: The next state produced, given (s,a).
            done: If the state is terminal. Normally 0 because non-episodic.

        ------
        """
        
        # Guardamos en memoria la tupla (s,a,r,s') + done #
        # Se guardan como arrays de numpy y al samplear se pasan a tensores #
        self.memory.store_transition(state, action, reward, next_state, done)        

    def sample_memory(self):

        """

        Extract 'self.batch_size' experiences (s,a,r,s') from the memory replay.

        ------

        Returns: The *BATCH_SIZE* (s,a,r,s') experiences.

        ------
        """
        
        # Muestreamos un conjunto BATCH de experiencias #
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        with T.no_grad():
            # Los convertimos a tensores de Pytorch #
            states = T.tensor(state, device = self.q_eval.device)
            rewards = T.tensor(reward, device = self.q_eval.device)
            dones = T.tensor(done, device = self.q_eval.device)
            actions = T.tensor(action, device = self.q_eval.device)
            next_state = T.tensor(new_state, device = self.q_eval.device)

        return states, actions, rewards, next_state, dones

    def prioritized_sample_memory(self):

        """

                Extract 'self.batch_size' experiences (s,a,r,s') from the memory replay.

                ------

                Returns: The *BATCH_SIZE* (s,a,r,s') experiences.

                ------
        """

        # Muestreamos un conjunto BATCH de experiencias #
        state, action, reward, new_state, done, indices, weight = self.memory.sample_buffer(self.batch_size, self.beta)

        with T.no_grad():
            # Los convertimos a tensores de Pytorch #
            states = T.tensor(state, device = self.q_eval.device)
            rewards = T.tensor(reward, device = self.q_eval.device)
            dones = T.tensor(done, device = self.q_eval.device)
            actions = T.tensor(action, device = self.q_eval.device)
            next_state = T.tensor(new_state, device = self.q_eval.device)
            weights = T.tensor(weight, device = self.q_eval.device)

        return states, actions, rewards, next_state, dones, indices, weights

    # Política e-greedy #
    def choose_action(self, observation, mode = 'egreedy'):

        """

        Epsilon-greedy policy. Plays explorate/explotate with a probability epsilon/(1-apsilon).

        ----

        Args:
            observation: The state (allegedly an image). Must be a matrix with (N_CHANN, HEIGHT, WIDTH)

        Returns: An action joint (1D array) with the selected actions.

        ------
        """
        
        if mode == 'egreedy':
            
            if np.random.random() > self.epsilon:
                
                with T.no_grad():
                    
                    state = T.tensor([observation], dtype=T.float, device = self.q_eval.device)
                    Q = self.q_eval.forward(state)
                    action_joint = []
                    
                    # En la e-greedy, si caemos en explotación, a = max_a(Q)
                    for i in range(self.action_joint_dim):
                        action_joint.append(T.argmax(Q.narrow(1,i*self.n_actions,self.n_actions)).item())
    
                return action_joint
            
            else:
                action_joint = np.random.choice(self.action_space, size = self.action_joint_dim)
                return action_joint
        
        elif mode == 'softmax':
            
            with T.no_grad():
                
                    state = T.tensor([observation], dtype=T.float, device = self.q_eval.device)
                    Q = self.q_eval.forward(state)
                    action_joint = []
                    
                    # Softmax policy #
                    for i in range(self.action_joint_dim):
                        probs = T.softmax(Q.narrow(1,i*self.n_actions,self.n_actions)/self.temperature,dim=1)
                        categ = T.distributions.Categorical(probs)
                        action_joint.append(categ.sample().item())
    
                    return action_joint
            
        else:
            assert('ERROR. ESCOJA UN VALOR DE POLÍTICA ADECUADO: (egreedy/softmax)')
    
    # Este método se usará out-class para poder alternar entre DDQN y DQN #
    def replace_target_network(self, epoch):
        """

        Function to dump the behavioral network into the target network.

        -----

        Args:
            epoch: The actual epoch.

        ------
        """
        """
        if epoch % self.update_target_count == 0:
            
            for target_param, param in zip(self.q_next.parameters(), self.q_eval.parameters()):
                
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        """
        
        if epoch % self.update_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        
        
    def learn(self, mask = None):
        """

        Learning function. It predicts the return (target_network) and takes a descent gradient step. The q-values are
        calculated with Time Difference and we will accumulate the other_results of the gradients along the agents.

        ------
        """

        # Si intentamos entrenar con menos experiencias que tamaño de batch
        # nos salimos y no entrenamos.
        if self.memory.mem_cntr < self.batch_size:
            return
        
        if mask is None:
            mask = np.zeros(shape=self.action_joint_dim)

        self.q_eval.optimizer.zero_grad()
        self.q_next.optimizer.zero_grad()

        if not self.prioritized:
            states, actions, rewards, next_states, dones = self.sample_memory()
        else:
            states, actions, rewards, next_states, dones, batches, weights = self.prioritized_sample_memory()
            prior = T.tensor(np.zeros(shape=batches.shape), device = self.q_eval.device)

        indices = np.arange(self.batch_size)
        
        Q_pred = self.q_eval(states)
        Q_next = self.q_next(next_states)
        Q_eval = self.q_eval(next_states)

        for i in range(self.action_joint_dim):
            
            if mask[i] == 1: # If the mask is 1, the agent does not learn at all #
                continue
            
            q_pred = Q_pred.narrow(1,i*self.n_actions,self.n_actions)[indices, actions[:, i]]
            max_actions = T.argmax(Q_eval.narrow(1,i*self.n_actions,self.n_actions), dim=1).detach()

            q_target = rewards[indices, i] + self.gamma*Q_next.narrow(1,i*self.n_actions,self.n_actions)[indices, max_actions[i]].detach() # TARGET IS DETACHED, ITS PARAMETERS ARE NOT SUBJECT OF TRAINING #
            
            if not self.prioritized:
                loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
                loss.backward(retain_graph=True)

            else:
                loss = self.q_eval.loss2(q_target, q_pred).to(self.q_eval.device)*weights
                prior += loss.data.detach()
                loss = loss.mean()
                loss.backward(retain_graph=True)

        self.q_eval.optimizer.step()

        if self.prioritized:
            self.memory.update_priorities(batches, prior.cpu().numpy())
        
    def decrement_epsilon(self):
        """
        Decrement 'self.epsilon' with a 'self.eps_dec', clipping its minimum to 'self.eps_min'.

        ------
        """
        
        if self.epsilon >= self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    def increment_beta(self):

        if self.beta >= 1:
            self.beta = 1
        else:
            self.beta += self.beta_increment
            
    def decrement_temperature(self):
        
        if self.temperature < 0.005:
            self.temperature = 0.005
        else:
            self.temperature = self.temperature - 2e-4
