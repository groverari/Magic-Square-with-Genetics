from magic_square import solve_magic_square, CrossoverMethod

# Constants
POPULATION_SIZE = 100
MUTATION_RATE = 0.5
ELITE_PERCENT = 0.8
NUM_GENERATIONS = 10000

# Both inclusive
MINIMUM_MAGIC_SQUARE_SIZE = 3
MAXIMUM_MAGIC_SQUARE_SIZE = 10

MAX_ATTEMPTS = 1

# CROSSOVER_METHOD = CrossoverMethod.FITNESS_PERCENTAGE
# CROSSOVER_METHOD = CrossoverMethod.SINGLE_CROSSOVER_POINT
# CROSSOVER_METHOD = CrossoverMethod.SINGLE_CROSSOVER_POINT_2
# CROSSOVER_METHOD = CrossoverMethod.DOUBLE_CROSSOVER_POINT
# CROSSOVER_METHOD = CrossoverMethod.DOUBLE_CROSSOVER_POINT_PERCENTAGE
CROSSOVER_METHOD = CrossoverMethod.UNIFORM_CROSSOVER

print("Population Size: " + str(POPULATION_SIZE))
print("Mutation Rate: " + str(MUTATION_RATE))
print("Elite Percent: " + str(ELITE_PERCENT))
print("Maximum Number of Generations: " + str(NUM_GENERATIONS))

print()

print("Minimum Magic Square Size: " + str(MINIMUM_MAGIC_SQUARE_SIZE))
print("Maximum Magic Square Size: " + str(MAXIMUM_MAGIC_SQUARE_SIZE))

print()

print("Maximum Attempts per Magic Square Size: " + str(MAX_ATTEMPTS))

print()

print("Crossover Method: " + str(CROSSOVER_METHOD))

print()

generations = {}
for magic_square_size in range(MINIMUM_MAGIC_SQUARE_SIZE, MAXIMUM_MAGIC_SQUARE_SIZE + 1):
    print("Solving for magic square size N=" + str(magic_square_size))
    for attempt_index in range(MAX_ATTEMPTS):
        generations[attempt_index] = []

        generation = solve_magic_square(square_size=magic_square_size, population_size=POPULATION_SIZE, num_generations=NUM_GENERATIONS, elite_percent=ELITE_PERCENT, mutation_rate=MUTATION_RATE, crossover_method=CROSSOVER_METHOD)
        if generation < NUM_GENERATIONS:
            print("Solution attempt " + str(attempt_index + 1) + " solved the magic square in " + str(generation) + " generations")
        else:
            print("Solution attempt " + str(attempt_index + 1) + " could not find a solution for magic square in " + str(generation) + " generations")

        generations[attempt_index].append(generation)

    if magic_square_size + 1 < MAXIMUM_MAGIC_SQUARE_SIZE + 1:
        print("")
