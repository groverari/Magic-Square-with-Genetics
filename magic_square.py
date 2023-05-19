import math
from statistics import mean
import numpy as np
import random
from crossover import *

# Calculates the square size, which is equal to the length of the magic_square array square-rooted.
def calculate_square_size(magic_square):
    return int(math.sqrt(len(magic_square)))

# Generate initial population
def generate_population(size, square_size):
    square_range = range(1, square_size**2 + 1)

    population = []
    for _ in range(size):
        population.append(random.sample(square_range, square_size**2))

    return population

# Calculates how incorrect every other row of the magic square is from the first, rather than the actual magic sum of the square.
def calculate_target_sum(square, square_size):
    return np.sum(square[0, :])  # Sum of the first row as the target sum

# Calculates the magic sum of a givens square size. Taken From 05/18/2023 2:44 PM: https://en.wikipedia.org/wiki/Magic_square?useskin=vector#Magic_constant
def calculate_true_target_sum(square, square_size):
    return square_size * (pow(square_size, 2) + 1) / 2

# Calculate fitness of an individual (lower fitness is better)
# np.trace finds the diagonals. flip square to get the second diagonal
# Calculates fitness by finding the sum of the absolute value of each row,column, and diagonal against the first row.
def calculate_fitness(individual, square_size):
    square = np.array(individual).reshape((square_size, square_size))

    # target_sum = calculate_target_sum(square=square, square_size=square_size)
    target_sum = calculate_true_target_sum(square=square, square_size=square_size)
    
    fitness = np.sum(np.abs(np.sum(square, axis=1) - target_sum)) + \
        np.sum(np.abs(np.sum(square, axis=0) - target_sum)) + \
        np.abs(np.trace(square) - target_sum) + \
        np.abs(np.trace(np.fliplr(square)) - target_sum)

    return fitness

# Perform mutation on an individual
# For every number, there is a chance this will swap 2 of the numbers in the array
def mutate(individual, mutation_rate):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

# For the entire square, there is a single chance that 2 of the numbers in the array will swap.
def single_mutate(individual, mutation_chance):
    if random.random() < mutation_chance:
        mutated = individual.copy()

        # TODO: Inefficient.
        randomized_individual = random.sample(individual, len(individual))
        
        from_index = randomized_individual.pop() - 1
        to_index = randomized_individual.pop() - 1

        mutated[from_index], mutated[to_index] = mutated[to_index], mutated[from_index]

        return mutated

    return individual

def guaranteed_mutate(individual, mutation_rate):
    mutated = individual.copy()

    mutation_amount = int(len(individual) * mutation_rate)

    randomized_individual = []
    for _ in range(mutation_amount):
        if len(randomized_individual) <= 1:
            # TODO: Inefficient way to do this.
            randomized_individual = random.sample(individual, len(individual))

        from_index = randomized_individual.pop() - 1
        to_index = randomized_individual.pop() - 1

        mutated[from_index], mutated[to_index] = mutated[to_index], mutated[from_index]

    return mutated

# Select individuals for the next generation using tournament selection
def selection(population, elite_percent, square_size, mutation_rate, crossover_method : CrossoverMethod, validate_children = True, random_fill_percentage = 0.0):
    fitness_scores = [calculate_fitness(
        individual, square_size) for individual in population]
    elite_count = int(len(population) * elite_percent)
    elite_indices = np.argsort(fitness_scores)[:elite_count]
    elites = [population[i] for i in elite_indices]
    offspring = elites.copy()

    while len(offspring) < len(population) * (1 - random_fill_percentage):
        parent1 = random.choice(elites)
        parent2 = random.choice(elites)

        # Use the provided crossover method or a random one if None.
        if crossover_method != None:
            child1, child2 = execute_crossover_method(parent1=parent1, parent2=parent2, crossover_method=crossover_method)
        else:
            child1, child2 = execute_crossover_method(parent1=parent1, parent2=parent2, crossover_method=random.choice(list(CrossoverMethod)))

        # This should be disabled once CrossoverMethod's are verified working.
        if validate_children:
            for i in range(1, len(parent1) + 1):
                assert i in child1
                assert i in child2

        # child1 = mutate(child1, mutation_rate)
        child1 = single_mutate(child1, mutation_rate)
        # child1 = guaranteed_mutate(child1, mutation_rate)
        
        # child2 = mutate(child2, mutation_rate)
        child2 = single_mutate(child2, mutation_rate)
        # child2 = guaranteed_mutate(child2, mutation_rate)

        # This should be disabled once CrossoverMethod's are verified working.
        if validate_children:
            for i in range(1, len(parent1) + 1):
                assert i in child1
                assert i in child2

        offspring.extend([child1, child2])

    offspring.extend(generate_population(size=len(population) - len(offspring), square_size=square_size))

    return offspring

# Solve the magic square problem using a genetic algorithm
def solve_magic_square(square_size, population_size, num_generations, elite_percent, mutation_rate, crossover_method : CrossoverMethod, validate_children = True, print_generation_updates = True):
    population = generate_population(population_size, square_size)
    generation = 0
    best_individual = min(
        population, key=lambda x: calculate_fitness(x, square_size))

    # stops loop at 20,000 generations or when the solution is found
    while (calculate_fitness(best_individual, square_size) > 0 and generation < num_generations):
        # if generation % 50000 != 0:
        #     population = selection(population=population, elite_percent=elite_percent, square_size=square_size, mutation_rate=mutation_rate, crossover_method=crossover_method, validate_children=validate_children, random_fill_percentage=0)
        # else:
            # population = selection(population=population, elite_percent=elite_percent, square_size=square_size, mutation_rate=mutation_rate, crossover_method=crossover_method, validate_children=validate_children, random_fill_percentage=0.2)

        population = selection(population=population, elite_percent=elite_percent, square_size=square_size, mutation_rate=mutation_rate, crossover_method=crossover_method, validate_children=validate_children, random_fill_percentage=0.1)

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
            average_fitness = mean([calculate_fitness(square, square_size) for square in population])

            print("Generation: " + str(generation) + " Best Fitness: " + str(calculate_fitness(best_individual, square_size)) + " Average Fitness: " + str(average_fitness))

    best_individual = min(
        population, key=lambda x: calculate_fitness(x, square_size))

    print("Best Individual: " + str(best_individual))

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
