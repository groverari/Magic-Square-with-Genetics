from enum import Enum, auto
import math
import numpy as np
import random

class CrossoverMethod(Enum):
    SINGLE_CROSSOVER_POINT = auto()
    SINGLE_CROSSOVER_POINT_2 = auto()
    DOUBLE_CROSSOVER_POINT = auto()
    DOUBLE_CROSSOVER_POINT_PERCENTAGE = auto()
    FITNESS_PERCENTAGE = auto()
    UNIFORM_CROSSOVER = auto()

# Calculates the square size, which is equal to the length of the magic_square array square-rooted.
def calculate_square_size(magic_square):
    return int(math.sqrt(len(magic_square)))

# Generate initial population
def generate_population(size, square_size):
    population = []
    for _ in range(size):
        individual = random.sample(
            range(1, square_size**2 + 1), square_size**2)
        population.append(individual)
    return population

# Calculate fitness of an individual (lower fitness is better)
# np.trace finds the diagonals. flip square to get the second diagonal
# Calculates fitness by finding the sum of the absolute value of each row,column, and diagonal against the first row.
def calculate_fitness(individual, square_size):
    square = np.array(individual).reshape((square_size, square_size))
    target_sum = np.sum(square[0, :])  # Sum of the first row as the target sum

    fitness = np.sum(np.abs(np.sum(square, axis=1) - target_sum)) + \
        np.sum(np.abs(np.sum(square, axis=0) - target_sum)) + \
        np.abs(np.trace(square) - target_sum) + \
        np.abs(np.trace(np.fliplr(square)) - target_sum)

    return fitness

# Calculates a crossover of two parents by setting the chance of picking the best (lower) fitness chance to better_fitness_percent, otherwise worse_fitness_percent.
def percentage_based_crossover(parent1, parent2, worse_fitness_percent = 0.1, better_fitness_percent = 0.9):
    square_size = calculate_square_size(parent1)

    parent1_chance = worse_fitness_percent
    # If the fitness of parent1 is less than (better) than the fitness of parent 2, set the chance of picking parent1 to the better fitness percentage, else it will be equal to the worse fitness percent chance.
    if worse_fitness_percent != better_fitness_percent and calculate_fitness(parent1, square_size) < calculate_fitness(parent2, square_size):
        parent1_chance = better_fitness_percent

    child1 = []
    child2 = []

    # For every child array, append the value at the current index point from parent1 if the random number from 0-1 is less than or equal to parent1 chance.
    # Otherwise, append from parent2.
    for child in [child1, child2]:
        for i in range(len(parent1)):
            if random.random() <= parent1_chance:
                child.append(parent1[i])
            else:
                child.append(parent2[i])

    return child1, child2

def uniform_crossover(parent1, parent2):
    return percentage_based_crossover(parent1=parent1, parent2=parent2, worse_fitness_percent=0.5, better_fitness_percent=0.5)

# Taken From 05/17/2023 3:14 PM: https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)?useskin=vector#Two-point_and_k-point_crossover
def double_point_crossover(parent1, parent2, crossover_point_min = 3, crossover_point_max_length_offset = -3):
    crossover_point_max = len(parent1) + crossover_point_max_length_offset

    child1 = []
    for i in range(len(parent1)):
        if i < crossover_point_min or i > crossover_point_max:
            child1.append(parent1[i])
        else:
            child1.append(parent2[i])

    child2 = []
    for i in range(len(parent2)):
        if i > crossover_point_min and i < crossover_point_max:
            child2.append(parent1[i])
        else:
            child2.append(parent2[i])

    return child1, child2

def double_point_crossover_percentage(parent1, parent2, crossover_point_min_percentage = 0.25, crossover_point_max_percentage = 0.75):
    crossover_point_min = int(len(parent1) * crossover_point_min_percentage)
    crossover_point_max = int(len(parent1) * crossover_point_max_percentage)

    return double_point_crossover(parent1=parent1, parent2=parent2, crossover_point_min=crossover_point_min, crossover_point_max_length_offset=crossover_point_max - len(parent1))

# Taken From 05/17/2023 3:05 PM: https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)?useskin=vector#One-point_crossover
# Seems to perform quite a bit better than the other single_point_crossover, even though they should be doing exactly the same thing.
def single_point_crossover_2(parent1, parent2, crossover_point_min = 1, crossover_point_max_length_offset = -1):
    # Pick a random point between the minimum crossover point and the length of the parents offset by the length offset, both inclusive.
    crossover_point = random.randint(crossover_point_min, len(parent1) + crossover_point_max_length_offset)

    child1 = []
    for i in range(len(parent1)):
        if i < crossover_point:
            child1.append(parent1[i])
        else:
            child1.append(parent2[i])

    child2 = []
    for i in range(len(parent2)):
        if i > crossover_point:
            child2.append(parent1[i])
        else:
            child2.append(parent2[i])

    return child1, child2

# Perform single-point crossover between two parents
# swaps numbers between two parents
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + \
        [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + \
        [gene for gene in parent1 if gene not in parent2[:crossover_point]]
    return child1, child2

# Perform mutation on an individual
# This swaps 2 of the numbers in the array
def mutate(individual, mutation_rate):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

# Select individuals for the next generation using tournament selection
def selection(population, elite_percent, square_size, mutation_rate, crossover_method : CrossoverMethod):
    fitness_scores = [calculate_fitness(
        individual, square_size) for individual in population]
    elite_count = int(len(population) * elite_percent)
    elite_indices = np.argsort(fitness_scores)[:elite_count]
    elites = [population[i] for i in elite_indices]
    offspring = elites.copy()

    while len(offspring) < len(population):
        parent1 = random.choice(elites)
        parent2 = random.choice(elites)

        if crossover_method == CrossoverMethod.SINGLE_CROSSOVER_POINT:
            child1, child2 = single_point_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.SINGLE_CROSSOVER_POINT_2:
            child1, child2 = single_point_crossover_2(parent1, parent2)
        elif crossover_method == CrossoverMethod.DOUBLE_CROSSOVER_POINT:
            child1, child2 = double_point_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.DOUBLE_CROSSOVER_POINT_PERCENTAGE:
            child1, child2 = double_point_crossover_percentage(parent1, parent2)
        elif crossover_method == CrossoverMethod.FITNESS_PERCENTAGE:
            child1, child2 = percentage_based_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.UNIFORM_CROSSOVER:
            child1, child2 = uniform_crossover(parent1, parent2)
        else:
            raise Exception("Unsupported CrossoverMethod passed to selection: " + str(crossover_method))

        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)

        offspring.extend([child1, child2])

    return offspring

# Solve the magic square problem using a genetic algorithm
def solve_magic_square(square_size, population_size, num_generations, elite_percent, mutation_rate, crossover_method : CrossoverMethod):
    population = generate_population(population_size, square_size)
    generation = 0
    best_individual = min(
        population, key=lambda x: calculate_fitness(x, square_size))

    # stops loop at 20,000 generations or when the solution is found
    while (calculate_fitness(best_individual, square_size) > 0 and generation < num_generations):
        population = selection(population=population, elite_percent=elite_percent, square_size=square_size, mutation_rate=mutation_rate, crossover_method=crossover_method)

        # Display the best individual in each generation
        best_individual = min(
            population, key=lambda x: calculate_fitness(x, square_size))
        """
        print(
            f"Generation {generation+1}: Best Fitness = {calculate_fitness(best_individual, square_size)}")
        print(np.array(best_individual).reshape((square_size, square_size)))
        print()
        """
        generation += 1

    best_individual = min(
        population, key=lambda x: calculate_fitness(x, square_size))

    # return np.array(best_individual).reshape((square_size, square_size))
    return generation

"""
THIS CODE RUNS THE MAGIC SQUARE IT IS COMMENTED OUT BECAUSE THE DATA FILE IS RUNNING THE FUNCTIONS
# Get the size of the square from the user
square_size = int(input("Enter the size of the square: "))

# Run the algorithm
best_solution = solve_magic_square(square_size)
print("Final Solution: \n", best_solution)
"""
