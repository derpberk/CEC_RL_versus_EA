import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
from Environment import Lake
import pickle


maps = []
importance_maps = []

# Load the different resolutions maps #
for i in range(4):
    maps.append(np.genfromtxt('map_{}.csv'.format(i+1), delimiter=','))
    importance_maps.append(np.genfromtxt('importance_map_{}.csv'.format(i+1), delimiter=','))

init_points = np.array([[5, 6], [11, 12], [17, 19], [23, 25]])


def evolute(cxpb = 0.8, mutpb = 0.2):

    hof_buffer = []
    logbook_buffer = []

    for r in range(len(maps)):

        # Creation of the environment #

        print(" ---- OPTIMIZING MAP NUMBER {} ----".format(r+1))

        env = Lake(filepath='map_{}.csv'.format(r+1),
                   number_of_agents = 1,
                   action_type="complete",
                   init_pos=init_points[r][np.newaxis],
                   importance_map_path='importance_map_{}.csv'.format(r+1))

        IND_SIZE = 8 # Number of actions #

        # Creation of the algorithm. Maximization like. #
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Generate a random action set

        toolbox.register("indices", np.random.randint, 0, 8, (r+1)*30)

        # Generación de inviduos y población
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, 10*(r+1)*30)

        # registro de operaciones genéticas
        toolbox.register("mate", tools.cxOrdered)
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

        random.seed(0)
        CXPB, MUTPB, NGEN = cxpb, mutpb, 100
        pop = toolbox.population()
        MU, LAMBDA = len(pop), len(pop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU,
                                                 LAMBDA, CXPB, MUTPB,
                                                 NGEN, stats=stats,
                                                 halloffame=hof)
        hof_buffer.append(hof)
        logbook_buffer.append(logbook)

    return hof_buffer, logbook_buffer


def plot_evolucion(log, i):

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
    plt.show()
    plt.savefig("EvolucionYpacarai_{}.png".format(i), dpi=300)


if __name__ == "__main__":

    v_best, v_log = evolute()
    print("Mejor fitness: %f" % v_best[0].keys[0].values[0])
    print("Mejor individuo %s" % v_best[0])

    with open('v_best', 'wb') as f:
        pickle.dump(v_best, f)
    with open('v_log', 'wb') as f:
        pickle.dump(v_log, f)

    for i in range(len(v_log)):
        plot_evolucion(v_log[i], i)
