import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
from Environment import Lake
import pickle
from copy import deepcopy
import argparse
import multiprocessing
from time import time

maps = []
importance_maps = []

# Load the different resolutions maps #
for i in range(4):
    maps.append(np.genfromtxt('map_{}.csv'.format(i+1), delimiter=','))
    importance_maps.append(np.genfromtxt('importance_map_{}.csv'.format(i+1), delimiter=','))

init_points = np.array([[5, 6], [11, 12], [17, 19], [23, 25]])

parser = argparse.ArgumentParser(description='Evolutionary computation of the .')
parser.add_argument('-R', metavar='R', type=int,
                    help='Resolution of the map', default=2)
parser.add_argument('--cxpb', metavar='cxpb', type=float,
                    help='Cross breed prob.', default=0.7)
parser.add_argument('--mutpb', metavar='mutpb', type=float,
                    help='Mut prob.', default=0.3)
args = parser.parse_args()

r = args.R
cxpb = args.cxpb
mutpb = args.mutpb

# Creation of the environment #

print(" ---- OPTIMIZING MAP NUMBER {} ----".format(r))

env = Lake(filepath='map_{}.csv'.format(r),
           number_of_agents=1,
           action_type="complete",
           init_pos=init_points[r - 1][np.newaxis],
           importance_map_path='importance_map_{}.csv'.format(r),
           num_of_moves=30*r)

IND_SIZE = 8  # Number of actions #

# Creation of the algorithm. Maximization like. #
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generate a random action set

toolbox.register("indices", np.random.randint, 0, 8, r * 30)

# Generación de inviduos y población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 10 * 30 * r)

# registro de operaciones genéticas
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def evalTrajectory(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""

    # distancia entre el último elemento y el primero
    env.reset()
    R = 0
    for t in range(len(individual)):
        _, reward = env.step([individual[t]])

        R += np.sum(reward)

    return R,

toolbox.register("evaluate", evalTrajectory)


def plot_evolucion(log, r):

    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    plt.plot(gen, fit_mins, "b")
    plt.plot(gen, fit_maxs, "r")
    plt.plot(gen, fit_ave, "--k")
    plt.fill_between(gen, fit_mins, fit_maxs,
                     facecolor="g", alpha=0.2)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend(["Min", "Max", "Avg"])
    plt.grid()
    plt.savefig("EvolucionYpacarai_{}.png".format(r), dpi=300)


def main():

    random.seed(0)
    CXPB, MUTPB, NGEN = cxpb, mutpb, 100 + 50 * (r-1)
    pop = toolbox.population()
    MU, LAMBDA = len(pop), len(pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    t0 = time()
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU,
                                             LAMBDA, CXPB, MUTPB,
                                             NGEN, stats=stats,
                                             halloffame=hof)
    print("He tardado {} segundos".format(time()-t0))
    return hof, logbook


if __name__ == "__main__":

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    hof, logbook = main()

    print("Mejor fitness: %f" % hof[0].fitness.values)
    print("Mejor individuo %s" % hof[0])

    with open('v_best_{}'.format(r), 'wb') as f:
        pickle.dump(hof, f)
    with open('v_log_{}'.format(r), 'wb') as f:
        pickle.dump(logbook, f)

