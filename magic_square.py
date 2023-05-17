import numpy as np
import random

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

# Perform single-point crossover between two parents
# swaps numbers between two parents
def crossover(parent1, parent2):
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
def selection(population, elite_size, square_size, mutation_rate):
    fitness_scores = [calculate_fitness(
        individual, square_size) for individual in population]
    elite_count = int(len(population) * elite_size)
    elite_indices = np.argsort(fitness_scores)[:elite_count]
    elites = [population[i] for i in elite_indices]
    offspring = elites.copy()

    while len(offspring) < len(population):
        parent1 = random.choice(elites)
        parent2 = random.choice(elites)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        offspring.extend([child1, child2])

    return offspring

# Solve the magic square problem using a genetic algorithm
def solve_magic_square(square_size, population_size, num_generations, elite_percent, mutation_rate):
    population = generate_population(population_size, square_size)
    generation = 0
    best_individual = min(
        population, key=lambda x: calculate_fitness(x, square_size))

    # stops loop at 20,000 generations or when the solution is found
    while (calculate_fitness(best_individual, square_size) > 0 and generation < num_generations):
        population = selection(population, elite_percent, square_size, mutation_rate)

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
