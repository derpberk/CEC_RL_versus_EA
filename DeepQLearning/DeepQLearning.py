
"""
    Training of a Deep Q-Learning for the Single-agent Ypacarai Case comparison with EA

"""
import sys

import numpy as np
from DDQNAgent import DDQNAgent
from Environment import Lake
import torch
from tqdm import trange
import signal
import argparse
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

""" Signal handler for graceful ending. """

def signal_handler(sig, frame):
    global end
    end = True
    print("\n ABORTING TRAINGING! \n")


if __name__ == "__main__":

    # Parsing arguments #
    parser = argparse.ArgumentParser(description='Ypacari Multi-agent Double Deep Q-Learning training.')
    parser.add_argument('-R', metavar='R', type=int,
                        help='Resolution', default=2)
    parser.add_argument('-E', metavar='E', type=int,
                        help='Number of epochs.', default=30000)

    init_points = np.array([[5, 6], [11, 12], [17, 19], [23, 25]])

    args = parser.parse_args()

    """ For graceful ending """
    end = False
    signal.signal(signal.SIGINT, signal_handler)
    
    """ Directory """
    now = datetime.now()
    directory = now.strftime("./results/TRAINING_DDQL_%d_%m_%Y_%H_%M_%S")
    os.makedirs(directory)
    
    """ For Logging """
    writer = SummaryWriter()
    
    # Create the evironment
    num_of_agents = args.N

    r = args.R
    env = Lake(filepath='map_{}.csv'.format(r),
               number_of_agents=1,
               action_type="complete",
               init_pos=init_points[r - 1][np.newaxis],
               importance_map_path='importance_map_{}.csv'.format(r),
               num_of_moves=30*r)

    """ --- HYPERPARAMETERS --- """

    gamma = 0.95
    num_of_epochs = args.E
    steps_per_episode = args.S
    epsilon = 0.99
    lr = 1e-4
    n_actions = 8
    input_dims = env.render.shape
    eps_min = 0.005
    eps_dec = (epsilon-eps_min)/(num_of_epochs*0.4)
  
    mem_size = 10000
    batch_size = 32
    update_target_count = 100
    action_joint_dim = num_of_agents
    prob_alpha = 0.6

    # For prioritized replay #
    prioritized = False
    beta = 0.4
    beta_increment = (1-beta)/(num_of_epochs*0.4)
    
    # For Softmax policy #
    temperature = 0.5
    
    # Soft updating #
    tau = 0.005


    """ Initialize numpy seed """
    np.random.seed(42)

    """ Creation of a multi-agent """

    multi_agent = DDQNAgent(
        gamma=gamma,
        epsilon=epsilon,
        lr=lr,
        n_actions=n_actions,
        input_dims=input_dims,
        action_joint_dim=action_joint_dim,
        mem_size=mem_size,
        batch_size=batch_size,
        eps_min=eps_min,
        eps_dec=eps_dec,
        replace=update_target_count,
        prioritized=prioritized,
        prob_alpha=prob_alpha, 
        beta=beta, 
        beta_increment=beta_increment,
        temperature = temperature,
        tau = tau

    )

    """ --- Buffers for results --- """
    filtered_reward_buff = []
    reward_buff = []
    record = -np.Inf  # Initial record #
    
    """ --- Empty Cuda Cache --- """
    
    torch.cuda.empty_cache()

    for epoch in trange(num_of_epochs):

        """ Reset the scenario. """
        env.reset()
        """ First state adquisition. """
        state = env.render()
        """ Episode reward """
        episode_reward = 0

        for step in range(steps_per_episode):

            # Choose an action #
            action = multi_agent.choose_action(state, mode = 'egreedy')

            # Apply the action joint #
            next_state, reward = env.step(action)

            # Store the experience #
            multi_agent.store_transition(state, action, reward, next_state, 0)

            # Update the state #
            state = next_state

            # Take a learning step #
            multi_agent.learn()

            # Accumulate episode reward #
            episode_reward += np.sum(reward)

        # Decrement epsilon #
        multi_agent.decrement_epsilon()

        # Update the target network (if needed) #
        multi_agent.replace_target_network(epoch)

        # Save the record network #
        if episode_reward > record:
            torch.save(multi_agent.q_eval, directory + '/BEST_Q_NETWORK_{}.pt'.format(r))
            print("New record of {}.".format(episode_reward))
            record = episode_reward
            np.savetxt(directory + '/iddle_mat_{}.csv'.format(r),env.idle_matrix, delimiter=',')
            np.savetxt(directory + '/visited_mat_{}.csv'.format(r), env.visited_matrix, delimiter = ',')

        """ ---- Data and results presentation ---- """

        reward_buff.append(episode_reward)
        writer.add_scalar('training/Acc_reward',episode_reward,epoch)

        if end:
            print("\n ---- PROCESS MANUALLY FINISHED IN EPOCH {} ----\n".format(epoch))
            break  # Finish the training


    """ Once the training is finished """

    print("FINISHED!")

    # Save the rewards
    np.savetxt(directory + '/reward_{}.csv'.format(r), reward_buff, delimiter=',')
    env.create_log()

    
    
    
    
    

