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

maps = []
importance_maps = []

init_points = np.array([[5, 6], [11, 12], [17, 19], [23, 25]])

parser = argparse.ArgumentParser(description='Evolutionary computation of the .')
parser.add_argument('-R', metavar='R', type=int,
                    help='Resolution of the map', default=2)
parser.add_argument('--cxpb', metavar='cxpb', type=float,
                    help='Cross breed prob.', default=0.7)
parser.add_argument('--mutpb', metavar='mutpb', type=float,
                    help='Mut prob.', default=0.3)

parser.add_argument('--recycle', metavar='recycle',
                    help='Recycle the population in the first case',default=None)

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

env2 = Lake(filepath='map_{}.csv'.format(r),
           number_of_agents=1,
           action_type="complete",
           init_pos=init_points[r - 1][np.newaxis],
           importance_map_path='alt_importance_map_{}.csv'.format(r),
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
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 10 * r * 30)

# registro de operaciones genéticas
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def evalTrajectory(individual):

    R = 0

    env.reset()
    for t in range(len(individual)):
        _, reward = env.step([individual[t]])

        R += np.sum(reward)

    return R,

def evalTrajectory2(individual):

    R = 0

    env2.reset()
    for t in range(len(individual)):
        _, reward = env2.step([individual[t]])

        R += np.sum(reward)

    return R,

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
    CXPB, MUTPB, NGEN = cxpb, mutpb, 100 + 50*(r-1)
    pop = toolbox.population()
    MU, LAMBDA = len(pop), len(pop)
    hof1 = tools.HallOfFame(1)
    hof2 = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()

    # First scenario #
    toolbox.register("evaluate", evalTrajectory)
    pop, logbook1 = algorithms.eaMuPlusLambda(pop, toolbox, MU,
                                             LAMBDA, CXPB, MUTPB,
                                             NGEN, stats=stats,
                                             halloffame=hof1)

    # Second scenario #
    toolbox.register("evaluate", evalTrajectory2)

    # If recycle flag is activated, re-initializate the population #
    if args.recycle is None:
        pop = toolbox.population()

    pop, logbook2 = algorithms.eaMuPlusLambda(pop, toolbox, MU,
                                             LAMBDA, CXPB, MUTPB,
                                             NGEN, stats=stats,
                                             halloffame=hof2)

    return (hof1,hof2), (logbook1,logbook2)


if __name__ == "__main__":

    pool = multiprocessing.Pool() # No args for the maximum number of processes
    toolbox.register("map", pool.map)

    if args.recycle is None:
        print("Recycling is not activated")
    else:
        print("Recycling is activated")

    hof, logbook = main()

    print("ESC1: Mejor fitness: %f" % hof[0][0].fitness.values)
    print("ESC1: Mejor individuo %s" % hof[0][0])
    print("ESC1: Mejor fitness: %f" % hof[1][0].fitness.values)
    print("ESC1: Mejor individuo %s" % hof[1][0])

    with open('generalization_v_best_ESC1_{}'.format(r), 'wb') as f:
        pickle.dump(hof[0], f)
    with open('generalization_v_best_ESC2_{}'.format(r), 'wb') as f:
        pickle.dump(hof[1], f)
    with open('generalization_v_log_ESC1_{}'.format(r), 'wb') as f:
        pickle.dump(logbook[0], f)
    with open('generalization_v_log_ESC2_{}'.format(r), 'wb') as f:
        pickle.dump(logbook[1], f)

