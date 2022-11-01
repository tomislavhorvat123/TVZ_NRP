#    This file is originally part of DEAP. Modified by Tomislav Horvat and Ivan Piščević.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
#
#   Changes made to suit the needs of LAB 1 for class NRP on TVZ.
#   - Individual class was changed from "set" to "list" type, to be able to hold duplicate items.
#   - evalKnapsack() function was modified to also calculate the number of duplicate items, and if
#     the number is too high to negatively score the Individual.
#   - removed cxSet() function which used to do crossover operation on given Individuals. Now it possible to use
#     built in (toolbox class) crossover functions.

import random
import time

import numpy
import pickle

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

IND_INIT_SIZE = 5   # Individual initial size -> toolbox.register("individual"....)
MAX_ITEM = 50       # Maximum number of items allowed inside individual
MAX_WEIGHT = 50     # Maximum weight allowed in Individual
NBR_ITEMS = 50      # Number of DIFFERENT ITEMS
MIN_ITEMS = 1       # Minimal number of items required in Individual
DUPLICATE_LIMIT = 3 # Number of repetitions allowed in Individual

# To assure reproducibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
# random.seed(64)

# Create the item dictionary: item name is an integer, and value is
# a (weight, value) 2-tuple.

items = {}
# MAX_WEIGHT of predefined item list is 60KG (if every single item is used 3 times)!
# items = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (4, 10), 5: (12, 4)}

# Create random items and store them in the items' dictionary.
# for i in range(NBR_ITEMS):
#     items[i] = (random.randint(1, 10), round(random.uniform(0, 100), 2))

a_file = open("data.pkl", "rb")
items = pickle.load(a_file)

# To have duplicates, Individual cannot be set, it must be a list.
creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Attribute generator - Returns a random integer from 0 to NBR_ITEMS - 1.
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
# toolbox.register("attr_item", random.randint, 1, NBR_ITEMS)

# Structure initializers - toolbox.attr_item is being called IND_INIT_SIZE times.
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    tooBigDuplicateFlag = False
    nonExistingItem = False

    for item in individual:

        if item not in items:
            nonExistingItem = True
            break
        else:
            if individual.count(item) > DUPLICATE_LIMIT:
                tooBigDuplicateFlag = True
                break
            weight += items[item][0]
            value += round(items[item][1], 2)

    if weight > MAX_WEIGHT:
        return 100, 0  # Ensure overweight bags are dominated

    if len(individual) > MAX_ITEM:
        return 200, 0  # Dominate if too many items in individual

    if len(individual) <= MIN_ITEMS:
        return 500, 0  # Dominate if less than allowed items

    if tooBigDuplicateFlag:
        return 200, 0  # Dominate if more than 3 items of same kind

    if nonExistingItem:
        return 1000, 0  # Dominate if individual contains non existing item

    return weight, value


def mutList(individual, mutPercentage):
    """Mutation that pops or adds an element."""

    if mutPercentage == 0:
        mutItemCount = 0
    elif mutPercentage > 1:
        mutItemCount = len(individual)
    else:
        mutItemCount = round(len(individual) * mutPercentage)

    for i in range(mutItemCount):
        if random.random() < 0.5:
            if len(individual) > 0:  # We cannot pop from an empty list
                individual.remove(random.choice(sorted(tuple(individual))))
        else:
            individual.append(random.randrange(NBR_ITEMS))
            # individual.append(random.randint(1, NBR_ITEMS))

    # if random.random() < 0.5:
    #     if len(individual) > 0:  # We cannot pop from an empty list
    #         individual.remove(random.choice(sorted(tuple(individual))))
    # else:
    #     individual.append(random.randrange(NBR_ITEMS))
    #     # individual.append(random.randint(1, NBR_ITEMS))

    return individual,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutList, mutPercentage=0.2)    #tools.mutGaussian, mu=0.5, sigma=1.2, indpb=0.2
toolbox.register("select", tools.selNSGA2)


def main():
    start_time = time.time()
    # random.seed(64)
    NGEN = 100          # NUMBER OF EVOLUTIONS
    MU = 50             # POPULATION SIZE
    LAMBDA = 100
    CXPB = 0.5
    MUTPB = 0.5

    pop = toolbox.population(n=MU)

    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    print("Vrijeme izvrsavanja: " + str(round(time.time() - start_time, 2)) + " s")
    record = stats.compile(pop)
    print(record)
    print(sorted(hof[-1]))
    return pop, stats, hof


if __name__ == "__main__":
    main()
