import numpy as np
from Environment import Lake


individual = [5, 3, 3, 7, 6, 7, 6, 7, 0, 7, 2, 0, 7, 3, 0, 3, 2, 2, 6, 6, 2, 7, 7, 1, 6, 5, 0, 2, 0, 3, 7, 4, 0, 4, 5,
              5, 6, 6, 3, 0, 7, 5, 0, 5, 1, 3, 4, 5, 2, 4, 7, 5, 7, 6, 0, 6, 3, 3, 2, 2, 2, 1, 6, 5, 0, 1, 2, 6, 1, 1,
              2, 6, 4, 1, 2, 2, 5, 5, 0, 5, 6, 3, 5, 2, 5, 0, 3, 7, 4, 7, 1, 1, 5, 1, 2, 5, 5, 5, 1, 1, 4, 2, 5, 5, 1,
              5, 6, 6, 5, 3, 3, 0, 7, 0, 0, 4, 6, 4, 3, 3]

individual = np.array(individual)

num_of_agents = 2
r = 2
init_points = np.array([[11, 12], [15, 14]])

env = Lake(filepath='map_{}.csv'.format(r),
           number_of_agents=1,
           action_type="complete",
           init_pos=init_points[0][np.newaxis],
           importance_map_path='importance_map_{}.csv'.format(r),
           num_of_moves=30*r)

def evalSingleAgentTrajectory(individual,i):
    """ Función objetivo, calcula la distancia que recorre el viajante"""

    # distancia entre el último elemento y el primero
    env.reset()
    R = 0
    for t in range(0, len(individual), num_of_agents):
        _, reward = env.step(individual[t+i][np.newaxis])

        R += np.sum(reward)

    return R,

for i in range(2):
    for p in range(2):

        env = Lake(filepath='map_{}.csv'.format(r),
                   number_of_agents=1,
                   action_type="complete",
                   init_pos=init_points[p][np.newaxis],
                   importance_map_path='importance_map_{}.csv'.format(r),
                   num_of_moves=30 * r)


        R = evalSingleAgentTrajectory(individual,i)

        print("Trayectoria del individuo i={} partiendo del punto p={}: R = {}".format(i,p,R))