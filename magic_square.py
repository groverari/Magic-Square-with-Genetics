from enum import Enum, auto
import math
import numpy as np
import random

class CrossoverMethod(Enum):
    SINGLE_CROSSOVER_POINT = auto()
    SEQUENTIAL_SEGMENT_CROSSOVER = auto()

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

# Taken From 05/18/2023 11:50 AM: https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Crossover_for_permutations
def sequential_segment_crossover(parent1, parent2, crossover_point_min = 4, crossover_point_max_length_offset = -4):
    crossover_point_max = len(parent1) + crossover_point_max_length_offset
    square_length = len(parent1)

    child1 = [0] * square_length
    for i in range(crossover_point_min, crossover_point_max + 1):
        child1[i] = parent1[i]

    # TODO: This is inefficient, as explicitly iterating the ranges which the crossover is not in will be faster than iterating all indices.
    for i in range(square_length):
        if child1[i] == 0:
            for value in parent2:
                if value not in child1:
                    child1[i] = value
                    break

    child2 = [0] * square_length
    for i in range(crossover_point_min, crossover_point_max + 1):
        child2[i] = parent2[i]

    # TODO: This is inefficient, as explicitly iterating the ranges which the crossover is not in will be faster than iterating all indices.
    for i in range(square_length):
        if child2[i] == 0:
            for value in parent1:
                if value not in child2:
                    child2[i] = value
                    break

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
def selection(population, elite_percent, square_size, mutation_rate, crossover_method : CrossoverMethod, validate_children = True):
    fitness_scores = [calculate_fitness(
        individual, square_size) for individual in population]
    elite_count = int(len(population) * elite_percent)
    elite_indices = np.argsort(fitness_scores)[:elite_count]
    elites = [population[i] for i in elite_indices]
    offspring = elites.copy()

    while len(offspring) < len(population):
        parent1 = random.choice(elites)
        parent2 = random.choice(elites)

        # Python has implicit breaks, so flow-through isn't a problem here.
        match crossover_method:
            case CrossoverMethod.SINGLE_CROSSOVER_POINT:
                child1, child2 = single_point_crossover(parent1, parent2)
            case CrossoverMethod.SEQUENTIAL_SEGMENT_CROSSOVER:
                child1, child2 = sequential_segment_crossover(parent1, parent2)
            case _:
                raise Exception("Unsupported CrossoverMethod passed to selection: " + str(crossover_method))

        # This should be disabled once CrossoverMethod's are verified working.
        if validate_children:
            for i in range(1, len(parent1) + 1):
                assert i in child1
                assert i in child2

        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)

        offspring.extend([child1, child2])

    return offspring

# Solve the magic square problem using a genetic algorithm
def solve_magic_square(square_size, population_size, num_generations, elite_percent, mutation_rate, crossover_method : CrossoverMethod, validate_children = True, print_generation_updates = True):
    population = generate_population(population_size, square_size)
    generation = 0
    best_individual = min(
        population, key=lambda x: calculate_fitness(x, square_size))

    # stops loop at 20,000 generations or when the solution is found
    while (calculate_fitness(best_individual, square_size) > 0 and generation < num_generations):
        population = selection(population=population, elite_percent=elite_percent, square_size=square_size, mutation_rate=mutation_rate, crossover_method=crossover_method, validate_children=validate_children)

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

        if print_generation_updates and generation % int(num_generations * 0.1) == 0:
            print("Generation: " + str(generation))

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
