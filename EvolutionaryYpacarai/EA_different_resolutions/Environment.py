#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Samuel Yanes Luis. University of Seville.

Class for the *Multi Agent Ypacarai* case and other lacustrine scenarios. PhD research.


"""

import numpy as np
import random
import os
import time

class Lake:
    
    """
    Parameters:
        `Lake.map`: Defines the surface of the legal zones for the drones to sail.

    -----

    Args:
        filepath: Path to the visitable/non-visitable Lake.map in a csv format.
        number_of_agents: Number of agent in the MARL scenario.
        action_type: Defines whether the agents can move diagonally or not.
        init_pos: Defines wether the initial position of the agents is provided or random (default)

    ![Render of the environment](./render_example.png)

    """
    
    def __init__(self, filepath, number_of_agents, importance_map_path=None,
                 action_type="complete", init_pos=None, fail_probability=0, num_of_moves = 30):

        assert os.path.isfile(filepath), "Invalid or corrupetd file in {}".format(filepath)
        
        self.map = np.genfromtxt(filepath, delimiter=',')  # Creation of the Lake.map attribute
        """ `filepath` atribute."""

        if importance_map_path is None:
            self.absolute_importance_map = np.copy(self.map)
        else:
            self.absolute_importance_map = np.genfromtxt(importance_map_path, delimiter=',')
        """ Defines the absolute importance of every zone. """

        self.absolute_importance_matrix = np.copy(self.absolute_importance_map).astype(float)
        """ Matrix with the static importance of every zone. """
        self.num_of_agents = number_of_agents  # Number of agents in the scenario
        """ Number of agents in the MARL case. """
        self.init_pos = np.copy(init_pos)

        # Properties of the map #
        
        self.visitable_zones = np.count_nonzero(self.map)
        """ Number of visitable cells. """

        self.num_of_zones = self.map.size
        """ Number of total cells. """

        self.nonvisitable_zones = self.num_of_zones - self.visitable_zones
        """ Number of obstacles. """

        self.height = self.map.shape[0]
        """ Height of the map. Number of rows. """

        self.width = self.map.shape[1]
        """ Width of the map. Number of columns. """

        self.timestep = 0
        """ Timesteps executed since the scenario initialization """
        
        assert self.num_of_agents < self.visitable_zones, "Too many agents! Try reducing the number of agents in play."

        # Reward function parameters #

        self.illegal_penalty = -5
        """ Penalty for an illegal action (visit an obstacle). """

        self.collision_penalty = -5
        """ Penalty for a collision. """

        self.new_cell_reward = 1
        """ Reward for visiting an undiscovered cell. """

        self.max_idleness = num_of_moves
        """ Maximum idleness for a cell. To normalize the reward. """
        
        self.min_importance = np.min(self.absolute_importance_map[self.map > 0])
        """ Minimum absolute importance in the map"""

        self.diagonal_penalty = -0.1
        """ Penalty for a diagonal movement. """
        
        self.number_of_collisions = 0
        self.number_of_illegals = 0

        # Agents properties #

        """ Defines the action space """
        self.action_type = action_type

        if self.action_type == "complete":
            self.action_space = 8  # All directions (with diagonals)
        else:
            self.action_space = 4  # Only cardinal points
            
        self.positions = np.zeros(shape=(self.num_of_agents, 2), dtype=int)  # Cartesian positions #
        """ Matrix with the positions of every agent.  """

        self.next_positions = np.zeros(shape=(self.num_of_agents, 2), dtype=int)
        """ Matrix with the attempted position of every agent given an acton joint. """

        self.fail_probability = fail_probability
        """ Probability of movement fail """

        # Agent batteries parameters #

        self.agents_batteries = np.ones(self.num_of_agents)*100
        """ Batteries for every agent in % """
        self.batteries_dev = 8
        """ Standard deviation of the initial battery"""
        self.batteries_mean = 85
        """ Mean of the initial battery"""
        self.batteries_min = 70
        """ Minimum initial battery"""
        self.batteries_max = 100
        """ Maximum initial battery"""
        self.are_batteries_simulated = False
        """ Trigger for batteries simulation """
        self.movement_battery_use = self.batteries_max/self.max_idleness
        """ Battery consumption for every non-diagonal movement """
        self.is_dead = np.zeros(self.num_of_agents)
        
        # Position of the agents #
        # Random positions. Must be changed in the future #

        if self.init_pos is None:
            i, j = np.nonzero(self.map)  # Select valid positions for agents (non-zeros).
            indx = random.sample(range(0, i.size), self.num_of_agents)  # Select random indexes for these positions.

            for n in range(self.num_of_agents):
                self.positions[n] = [i[indx[n]], j[indx[n]]]  # Initialize this positions.
        else:
            
            selected = np.random.choice(len(self.init_pos),self.num_of_agents,replace = False)
            self.positions = np.copy(self.init_pos[selected])
            
        # Visitation matrixes 
        self.visited_matrix = np.zeros(shape=self.map.shape, dtype=int)
        """ A global visitation matrix. 1 for visited and 0 for unvisited."""
        self.idle_matrix = np.ones(shape=self.map.shape, dtype=int)*self.max_idleness
        """ A idle time matrix. Each position represents the number f timesteps the cell 
        has remain unvisited since the last visit. """
        self.absolute_importance_substraction = 0.2

        # Mark the first positions as visited #
        for n in range(self.num_of_agents):
            self.visited_matrix[int(self.positions[n][0]), int(self.positions[n][1])] = 1
            
        # Other atributes #
        
        self.outside_colors = np.asarray([0.4, 0.2, 0.2])
        """ Color of the borders """
        self.colors = ((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(1.0,0.0,1.0),(1.0,1.0,0.0),(0.0,1.0,1.0))
        """ List of colors for the agents. """
        self.dead_color = np.asarray([0.8, 0.3, 0.8])
        
        self.init_time = time.monotonic()

    def simulate_batteries(self, trigg=True):
        """
            Function to simulate the batteries of every agent.

        -----

        Args:
            trigg: flag to activate or deactivate the battery simulation.

        """

        self.are_batteries_simulated = trigg

    def init_batteries(self):
        """
            Method to initialize the battery levels of every agent with a random value. This value responds to a normal
            distribution with a mean value, a std. deviation and a minimum and maximum (100%) value.

        """

        """ Reset the status of every battery. """
        possible_levels = np.linspace(self.batteries_max,self.batteries_min,self.num_of_agents)
        
        self.agents_batteries = np.random.choice(possible_levels, size=self.num_of_agents, replace = False)
        """ Reset the status of every drone. """
        self.is_dead = np.zeros(self.num_of_agents)

    def use_battery(self, action_joint):

        """
        Args:
            `action_joint`: Receives the action joint to compute the use of the battery depending on the movement.

        ![Example of dead ASV battery](./dead_battery_example.png)

        """

        if self.are_batteries_simulated: # If the batteries are simulated...

            for n in range(self.num_of_agents):

                if self.agents_batteries[n] >= self.movement_battery_use: # If it is possible to move...

                    self.agents_batteries[n] -= self.movement_battery_use
                    
                else:
                    self.is_dead[n] = 1  # If it is not possible, declare the agent DEAD!
                

    def random_action(self):
        """
            Generates a random joint action.

            -----

            Returns:
                action_joint: List with the action joint.
        """
        action_joint = np.random.randint(self.action_space, size=self.num_of_agents)
        
        return action_joint
    
    def reset(self):
        """
            Reset the scenario and clear the matrices.

        """

        self.timestep = 0
        self.init_batteries()
        self.number_of_collisions = 0
        self.number_of_illegals = 0

        """ Reset the timesteps """

        # Reposition of the agents #
        # Random positions or init_pos provided #

        if self.init_pos is None:
            i, j = np.nonzero(self.map)  # Select valid positions for agents (non-zeros).
            indx = random.sample(range(0, i.size), self.num_of_agents)  # Select random indexes for these positions.

            for n in range(self.num_of_agents):
                self.positions[n] = [i[indx[n]], j[indx[n]]]  # Initialize this positions.
        else:
            selected = np.random.choice(len(self.init_pos),self.num_of_agents, replace = False)
            self.positions = np.copy(self.init_pos[selected])
            
        # Visitation matrixes 
        self.visited_matrix = np.zeros(shape=self.map.shape, dtype=int)
        self.idle_matrix = np.ones(shape=self.map.shape, dtype=int)*self.max_idleness
        self.absolute_importance_matrix = np.copy(self.absolute_importance_map).astype(float)
        
        for n in range(self.num_of_agents):
            self.visited_matrix[self.positions[n, 0], self.positions[n, 1]] = 1
         
    def step(self, action_joint):
        """
            Process an action joint list and calculate the collisions due to this action.

            -----

            Args:
                action_joint: List with the action joint.

            -----

            Returns:
                state: Returns the rendered scenario as a 3-channel (RGB) image.
                reward: A 1D array with the reward for every agent.

        """

        # Augment the state count #
        self.timestep += 1
        
        # We calculate the intended_position array #
        intended_position = np.zeros(shape=(self.num_of_agents, 2))

        # Compute posible movements fails #
        fail_table = self.movement_fail(p=self.fail_probability)

        for n in range(self.num_of_agents):  # Iteramos para cada agente.

            # Compute possible movement fails #
            if not fail_table[n]:
                vector = self.action_to_vector(action_joint[n])
            else:
                vector = self.action_to_vector(np.random.randint(self.num_of_agents))

            # Check if is battery-dead #
            if self.is_dead[n] == 0:
                intended_position[n] = self.positions[n] + vector
            else:
                intended_position[n] = self.positions[n]

        # Once calculated, we check if the movement is posible #
        
        illegal_table, collision_table = self.verify_and_update_movement(intended_position)

        """ Compute the battery utilization for every agent """
        self.use_battery(action_joint)

        """ We compute the idleness each agent encounter when visiting the new position. """
        idleness_table = np.zeros(self.num_of_agents)
        for n in range(self.num_of_agents):
            idleness_table[n] = self.idle_matrix[self.positions[n, 0], self.positions[n, 1]]

        reward = self.reward_function(idleness_table, illegal_table, collision_table, action_joint)

        self.idle_matrix = np.clip(self.idle_matrix+1, 0, self.max_idleness)  # Add 1 to the idleness of every zone and clip

        """ Update the scenario with the acceptable movements. """
        for n in range(self.num_of_agents):
            
            if self.is_dead[n] == 0:
                # Update the visited zones with the new positions
                self.visited_matrix[self.positions[n, 0], self.positions[n, 1]] += 1
                # Substract the absolute importance of the visited zone
                self.absolute_importance_matrix[self.positions[n, 0], self.positions[n, 1]] -= self.absolute_importance_map[self.positions[n, 0], self.positions[n, 1]]*self.absolute_importance_substraction
            
            # Reset the idleness in the position of the agents
            self.idle_matrix[self.positions[n, 0], self.positions[n, 1]] = 0

                

        """ Render the scenario """
        state = self.render()

        return state, reward

    @staticmethod
    def action_to_vector(action):

        """
            Receives a single action and returns a vector with the positional traslation.

            -----

            Args:
                action: Single action.

            -----

            Returns:
                1D list with the traslational vector for the given action.

        """

        if action == 0:  # NORTH
            return [-1, 0]
        elif action == 1:  # EAST
            return [0, 1]
        elif action == 2:  # SOUTH
            return [1, 0]
        elif action == 3:  # WEST
            return [0, -1]
        elif action == 4:  # NORTH-EAST
            return [-1, 1]
        elif action == 5:  # SOUTH-EAST
            return [1, 1]
        elif action == 6:  # SOUTH-WEST
            return [1, -1]
        elif action == 7:  # NORTH-WEST
            return [-1, -1]

    def verify_and_update_movement(self, intended_position):
        """
            Receives the intended positions of every agent and computes if the movement other_results in a illegal move or
            in a collision between agents. Then update the position of every agent.

            -----

            Args:
                intended_position: Position matrix of the new position each agent tries to reach.

            -----

            Returns:
                illegal_table: 1D array with the agents that intended to move to an illegal cell.
                collision_table: 1D array with the agents that have collisioned due to their actions.

        """
        collision_table = np.zeros(self.num_of_agents)
        """ Verification of which agent collides. An 1 indicates a collision between agents. """
        illegal_table = np.zeros(self.num_of_agents)
        """ Verification of which agent scape from visitable zones. An 1 indicates an illegal attempt. """
        
        for n in range(self.num_of_agents):
            
            i = int(intended_position[n, 0])
            j = int(intended_position[n, 1])

            """ Verification of illegal movements. """

            if self.map[i, j] == 0:
                illegal_table[n] = 1
                intended_position[n] = np.copy(self.positions[n]) # If ille

        for n in range(self.num_of_agents):
            for n in range(self.num_of_agents):
                for k in range(n+1, self.num_of_agents):
    
                    # Same-place collision
                    if (intended_position[n] == intended_position[k]).all():
                        collision_table[n] = 1
                        collision_table[k] = 1
                        intended_position[k] = np.copy(self.positions[k])
                        intended_position[n] = np.copy(self.positions[n])
    
                    # Switching-position collision
                    if (intended_position[n] == self.positions[k]).all() \
                            and (self.positions[n] == intended_position[k]).all():
                        collision_table[n] = 1
                        collision_table[k] = 1
                        intended_position[k] = np.copy(self.positions[k])
                        intended_position[n] = np.copy(self.positions[n])
    
                    if (intended_position[n] == self.positions[k]).all() and (collision_table[k] == 1 or illegal_table[k] == 1):
                        collision_table[n] = 1
                        collision_table[k] = 1
                        intended_position[k] = np.copy(self.positions[k])
                        intended_position[n] = np.copy(self.positions[n])

        for n in range(self.num_of_agents):
            self.positions[n] = np.copy(intended_position[n])
        
        self.number_of_collisions += np.sum(collision_table/2)
        self.number_of_illegals += np.sum(illegal_table)
        
        return illegal_table, collision_table

    def render(self, agent=-1):
        """
            Compute the render of the environment.

            Args:
                agent: the index of the agent for whom we want to specifically render the scenario. If none agent is
                selected, a complete, general and non-specific render is returned.

            -----

            Returns:
                RGB: a 3-channel RGB matrix with the render of the scenario.


        """

        rgb = np.zeros(shape=(4, self.height, self.width))
        """ 3-channel image for rendering. """

        for i in range(self.height):
            for j in range(self.width):
                
                if self.map[i, j] == 0:
                    rgb[0:3, i, j] = self.outside_colors
                else:
                    """
                    if self.visited_matrix[i, j] == 0:
                        rgb[0:3, i, j] = np.asarray([0.0, 0.0, 0.0])
                    else:
                        rgb[0:3, i, j] = np.asarray([1.0, 1.0, 1.0]) - np.asarray([1.0, 1.0, 1.0]) * (self.idle_matrix[i, j]/self.max_idleness * np.clip(self.absolute_importance_matrix[i, j],0,1))
                    """
                    rgb[0:3, i, j] = np.asarray([1.0, 1.0, 1.0]) - np.asarray([1.0, 1.0, 1.0]) * self.idle_matrix[i, j]/self.max_idleness * np.clip(self.absolute_importance_matrix[i, j],0,1)
   
        for n in range(self.num_of_agents):
            
            i = self.positions[n][0]
            j = self.positions[n][1]
            rgb[3,i,j] = 1
            
            if agent == -1:
                rgb[0:3,i, j] = list(self.colors[n % 10])
            else:
                if agent == n:
                    rgb[0:3,i, j] = np.asarray([1.0, 0.0, 0])
                else:
                    rgb[0:3,i, j] = np.asarray([0.0, 0.0, 0.5])

            if self.is_dead[n]:
                rgb[0:3, i, j] = np.asarray(self.dead_color)

        return rgb

    def reward_function(self, idleness_table, illegal_table, collision_table, action_joint):

        """
            Receives `idleness_table`, `illegal_table`, `collision_table` and `action_joint` and computes the reward
            table for each agent.

            -----

            Args:
                idleness_table: Table with the idleness of the cells captured by each agent.
                illegal_table: Table with the agents that will be punished for visiting an illegal cell.
                collision_table: Table with the agents with collisions.
                action_joint: The 1D array with the action id for diagonal movement penalization.

            -----

            Returns:
                reward: Table with the reward for every agent.

        """
        reward = np.zeros(self.num_of_agents)

        for n in range(self.num_of_agents):

            if action_joint[n] >= 4:  # If diagonal movement exists, a penalization is applied.
                reward[n] += self.diagonal_penalty

            if self.is_dead[n] == 1:
                reward[n] = 0
            elif illegal_table[n] == 1:
                reward[n] += self.illegal_penalty  # Penalty for illegal movement.
            elif collision_table[n] == 1:
                reward[n] += self.collision_penalty  # Penalty for collision.
            elif self.visited_matrix[self.positions[n, 0], self.positions[n, 1]] == 0:  # If cell not visited before...
                reward[n] += self.new_cell_reward * self.absolute_importance_matrix[self.positions[n, 0], self.positions[n, 1]]
                # The reward is the maximum reward times the absolute importance of the visited cell.
            else:  # If legal, not collision and already visited...
                reward[n] += idleness_table[n]/self.max_idleness * self.absolute_importance_matrix[self.positions[n, 0], self.positions[n, 1]]

        return reward

    def movement_fail(self, p=0.05, distribution='constant'):

        """
            Receives `p` and a 'distribution' name. Computes a vector of boolean values. A True in a position i means
            the movement of this specific (i) agent fails and perform a different movement from the indicated originally.

            -----

            Args:
                p: Probability of fail.
                distribution: Probability distribution. Only implemented a constant probability dist. at the moment.

            -----

            Returns:
                reward: 1D booleans array with a fail condition.

        """

        if distribution == 'constant':
            return np.random.rand(self.num_of_agents) < p
        else:
            print("Select a valid probability distribution!")
            return np.zeros(self.num_of_agents) == 0

    def create_log(self, pathfile='.'):

        """
            Creates a log with the condition of the simulation in order to track the conditions of every experiment.

        """
        import datetime

        attributes = self.__dict__
        date = datetime.datetime.now()

        f = open(pathfile + "/" + date.strftime("%c").replace(" ","_") + "_LOGFILE.log", "w")
        f.write("Simulacion del entorno multiagente\n")
        f.write("----------- ATRIBUTOS DEL EXPERIMENTO ------------\n")
        f.write("\n")

        for key, value in attributes.items():
            f.write("\n--------- ")
            f.write(key + ":\n")
            f.write(str(value))
            f.write("\n")
            
        f.write("\n--------- ")
        f.write('ELAPSED_TIME' + ":\n")
        f.write(str(time.monotonic() - self.init_time) + ' seconds')
        f.write("\n")
            
        f.close()

        print("LOG CREADO EN " + pathfile)

    def safe_step(self,action_joint):
        """
            Process a SAFE action joint list

            -----

            Args:
                action_joint: List with the action joint.

            -----

            Returns:
                state: Returns the rendered scenario as a 3-channel (RGB) image.
                reward: A 1D array with the reward for every agent.

        """

        # Augment the state count #
        self.timestep += 1
        
        # We calculate the intended_position array #
        intended_position = np.zeros(shape=(self.num_of_agents, 2))

        # Compute posible movements fails #
        fail_table = self.movement_fail(p=self.fail_probability)
        
        safe = False
        
        while safe is False:

            for n in range(self.num_of_agents):  # Iteramos para cada agente.
    
                # Compute possible movement fails #
                if not fail_table[n]:
                    vector = self.action_to_vector(action_joint[n])
                else:
                    vector = self.action_to_vector(np.random.randint(self.num_of_agents))
    
                # Check if is battery-dead #
                if self.is_dead[n] == 0:
                    intended_position[n] = self.positions[n] + vector
                else:
                    intended_position[n] = self.positions[n]
    
            # Once calculated, we check if the movement is posible #
            
            illegal_table, collision_table = self.verify_movement(intended_position)
            
            if np.sum(illegal_table) == 0 and np.sum(collision_table) == 0 :
                
                safe = True
                
            else:
                
                for n in range(self.num_of_agents):
                    
                    if collision_table[n] > 0 or illegal_table[n] > 0:
                        
                        action_joint[n] = np.random.randint(0,high = self.action_space)
                        
        illegal_table, collision_table = self.verify_and_update_movement(intended_position)
            
        """ Compute the battery utilization for every agent """
        self.use_battery(action_joint)

        """ We compute the idleness each agent encounter when visiting the new position. """
        idleness_table = np.zeros(self.num_of_agents)
        for n in range(self.num_of_agents):
            idleness_table[n] = self.idle_matrix[self.positions[n, 0], self.positions[n, 1]]

        reward = self.reward_function(idleness_table, illegal_table, collision_table, action_joint)

        self.idle_matrix = np.clip(self.idle_matrix+1, 0, self.max_idleness)  # Add 1 to the idleness of every zone and clip

        """ Update the scenario with the acceptable movements. """
        for n in range(self.num_of_agents):
            # Update the visited zones with the new positions
            self.visited_matrix[self.positions[n, 0], self.positions[n, 1]] += 1
            # Reset the idleness in the position of the agents
            self.idle_matrix[self.positions[n, 0], self.positions[n, 1]] = 0
            # Substract the absolute importance of the visited zone
            self.absolute_importance_matrix[self.positions[n, 0], self.positions[n, 1]] -= self.absolute_importance_map[self.positions[n, 0], self.positions[n, 1]]*self.absolute_importance_substraction

        """ Render the scenario """
        state = self.render()

        return state, reward, action_joint
 
    def verify_movement(self, intended_position):
        """
            Receives the intended positions of every agent and computes if the movement other_results in a illegal move or
            in a collision between agents. Then update the position of every agent.

            -----

            Args:
                intended_position: Position matrix of the new position each agent tries to reach.

            -----

            Returns:
                illegal_table: 1D array with the agents that intended to move to an illegal cell.
                collision_table: 1D array with the agents that have collisioned due to their actions.

        """
        collision_table = np.zeros(self.num_of_agents)
        """ Verification of which agent collides. An 1 indicates a collision between agents. """
        illegal_table = np.zeros(self.num_of_agents)
        """ Verification of which agent scape from visitable zones. An 1 indicates an illegal attempt. """
        
        for n in range(self.num_of_agents):
            
            i = int(intended_position[n, 0])
            j = int(intended_position[n, 1])

            """ Verification of illegal movements. """

            if self.map[i, j] == 0:
                illegal_table[n] = 1
                intended_position[n] = np.copy(self.positions[n]) # If ille

        for n in range(self.num_of_agents):
            for n in range(self.num_of_agents):
                for k in range(n+1, self.num_of_agents):
    
                    # Same-place collision
                    if (intended_position[n] == intended_position[k]).all():
                        collision_table[n] = 1
                        collision_table[k] = 1
                        intended_position[k] = np.copy(self.positions[k])
                        intended_position[n] = np.copy(self.positions[n])
    
                    # Switching-position collision
                    if (intended_position[n] == self.positions[k]).all() \
                            and (self.positions[n] == intended_position[k]).all():
                        collision_table[n] = 1
                        collision_table[k] = 1
                        intended_position[k] = np.copy(self.positions[k])
                        intended_position[n] = np.copy(self.positions[n])
    
                    if (intended_position[n] == self.positions[k]).all() and (collision_table[k] == 1 or illegal_table[k] == 1):
                        collision_table[n] = 1
                        collision_table[k] = 1
                        intended_position[k] = np.copy(self.positions[k])
                        intended_position[n] = np.copy(self.positions[n])
        
        self.number_of_collisions += np.sum(collision_table/2)
        self.number_of_illegals += np.sum(illegal_table)
        
        return illegal_table, collision_table          