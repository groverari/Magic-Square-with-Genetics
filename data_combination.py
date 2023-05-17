from magic_square import solve_magic_square

# Constants
MUTATION_RATE = 0.1
ELITE_PERCENT = 0.1
NUM_GENERATIONS = 2000

# Both inclusive
MINIMUM_MAGIC_SQUARE_SIZE = 3
MAXIMUM_MAGIC_SQUARE_SIZE = 10

MAX_ATTEMPTS = 5

generations = {}
for magic_square_size in range(MINIMUM_MAGIC_SQUARE_SIZE, MAXIMUM_MAGIC_SQUARE_SIZE + 1):
    print("Solving for magic square size N=" + str(magic_square_size))
    for attempt_index in range(MAX_ATTEMPTS):
        generations[attempt_index] = []

        generation = solve_magic_square(magic_square_size, NUM_GENERATIONS, ELITE_PERCENT, MUTATION_RATE)
        if generation < NUM_GENERATIONS:
            print("Solution attempt " + str(attempt_index + 1) + " solved the magic square in " + str(generation) + " generations")
        else:
            print("Solution attempt " + str(attempt_index + 1) + " could not find a solution for magic square in " + str(generation) + " generations")

        generations[attempt_index].append(generation)

    if magic_square_size + 1 < MAXIMUM_MAGIC_SQUARE_SIZE + 1:
        print("")
