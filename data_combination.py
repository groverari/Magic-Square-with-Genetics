import random
import statistics
import time
import numpy
from typing import List
from magic_square import solve_magic_square
from crossover import CrossoverMethod

# Fix the seed in place for both Python and numpy.
random.seed(0)
numpy.random.seed(0)

# Constants
POPULATION_SIZE = 200
MUTATION_RATE = 0.5
ELITE_PERCENT = 0.8
NUM_GENERATIONS = 10000

# Both inclusive
MINIMUM_MAGIC_SQUARE_SIZE = 3
MAXIMUM_MAGIC_SQUARE_SIZE = 10

MAX_ATTEMPTS = 3

CROSSOVER_METHOD = CrossoverMethod.SINGLE_CROSSOVER_POINT
# CROSSOVER_METHOD = CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE

# CROSSOVER_METHOD = CrossoverMethod.SEQUENTIAL_SEGMENT_CROSSOVER
# CROSSOVER_METHOD = CrossoverMethod.SEQUENTIAL_SEGMENT_CROSSOVER_PERCENTAGE

# CROSSOVER_METHOD = CrossoverMethod.RANDOM_SEGMENT_CROSSOVER
# CROSSOVER_METHOD = CrossoverMethod.RANDOM_SEGMENT_CROSSOVER_PERCENTAGE

# CROSSOVER_METHOD = CrossoverMethod.ORDER_CROSSOVER

# CROSSOVER_METHOD = CrossoverMethod.VOTING_RECOMBINATION

# CROSSOVER_METHOD = CrossoverMethod.ALTERNATING_POSITION_CROSSOVER

# Will choose a random crossover method each time.
# CROSSOVER_METHOD = None

# Determines whether children should be validated for having one of every value from 1 to the length of the magic square.
# This is somewhat expensive, so it should only be done when a new CrossoverMethod is added.
VALIDATE_CHILDREN = True

output_to_csv = True
generate_test_cases = False

def to_titlecase(string : str):
    return string.replace("_", " ").title().strip()

class MagicSquareTestCase:
    def __init__(self, population_size : int, square_size : int, mutation_rate : float, elite_percent : float, crossover_method : CrossoverMethod):
        self.population_size = population_size
        self.square_size = square_size
        self.mutation_rate = mutation_rate
        self.elite_percent = elite_percent
        self.crossover_method = crossover_method

    def __str__(self):
        return to_titlecase(str(self.crossover_method).replace("CrossoverMethod.", "").replace("CROSSOVER", "")) + "," + str(self.population_size) + "," + str(self.mutation_rate) + "," + str(self.elite_percent) + "," + str(self.square_size)

# Contains all of the test cases to be ran by the program.
magic_square_test_cases : List[MagicSquareTestCase] = [
    MagicSquareTestCase(population_size=10, square_size=3, mutation_rate=0.8, elite_percent=0.2, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE),
    MagicSquareTestCase(population_size=10, square_size=5, mutation_rate=0.8, elite_percent=0.2, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE),
    MagicSquareTestCase(population_size=10, square_size=10, mutation_rate=0.8, elite_percent=0.2, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE),

    # MagicSquareTestCase(population_size=1000, square_size=3, mutation_rate=0.4, elite_percent=0.2, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT),
    # MagicSquareTestCase(population_size=1000, square_size=4, mutation_rate=0.4, elite_percent=0.2, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT),
    # MagicSquareTestCase(population_size=1000, square_size=5, mutation_rate=0.4, elite_percent=0.2, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT),

    MagicSquareTestCase(population_size=100, square_size=3, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE),
    MagicSquareTestCase(population_size=100, square_size=5, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE),
    MagicSquareTestCase(population_size=100, square_size=10, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE),

    MagicSquareTestCase(population_size=100, square_size=3, mutation_rate=0.8, elite_percent=0.2, crossover_method=CrossoverMethod.RANDOM_SEGMENT_CROSSOVER_PERCENTAGE),
    MagicSquareTestCase(population_size=100, square_size=5, mutation_rate=0.8, elite_percent=0.2, crossover_method=CrossoverMethod.RANDOM_SEGMENT_CROSSOVER_PERCENTAGE),
    MagicSquareTestCase(population_size=100, square_size=10, mutation_rate=0.8, elite_percent=0.2, crossover_method=CrossoverMethod.RANDOM_SEGMENT_CROSSOVER_PERCENTAGE),

    MagicSquareTestCase(population_size=100, square_size=3, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.ORDER_CROSSOVER),
    MagicSquareTestCase(population_size=100, square_size=5, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.ORDER_CROSSOVER),
    MagicSquareTestCase(population_size=100, square_size=10, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.ORDER_CROSSOVER),

    MagicSquareTestCase(population_size=100, square_size=3, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.VOTING_RECOMBINATION),
    MagicSquareTestCase(population_size=100, square_size=5, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.VOTING_RECOMBINATION),
    MagicSquareTestCase(population_size=100, square_size=10, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.VOTING_RECOMBINATION),

    MagicSquareTestCase(population_size=100, square_size=3, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.ALTERNATING_POSITION_CROSSOVER),
    MagicSquareTestCase(population_size=100, square_size=5, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.ALTERNATING_POSITION_CROSSOVER),
    MagicSquareTestCase(population_size=100, square_size=10, mutation_rate=0.5, elite_percent=0.8, crossover_method=CrossoverMethod.ALTERNATING_POSITION_CROSSOVER),
]

# If we need to generate the test cases, we append the above list with the permutations of the various constant parameters at the top of the file.
if generate_test_cases:
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

    for magic_square_size in range(MINIMUM_MAGIC_SQUARE_SIZE, MAXIMUM_MAGIC_SQUARE_SIZE + 1):
        magic_square_test_cases.append(MagicSquareTestCase(population_size=POPULATION_SIZE, square_size=magic_square_size, mutation_rate=MUTATION_RATE, elite_percent=ELITE_PERCENT, crossover_method=CROSSOVER_METHOD))

csv_lines : List[str] = []

# Tests every test case, printing the perfroamcne and the generations it took to find a solution.
# TODO: Could very easily be multithreaded, increases training test time by a significant amount.
for magic_square_test_case in magic_square_test_cases:
    print("Running test case " + str(magic_square_test_case))
    generations : List[int] = []

    dnfs = 0
    for attempt_index in range(MAX_ATTEMPTS):
        # Solve for the magic square with the parameters of this test case, recording the time it took to execute the function.
        pre_time = time.perf_counter_ns()
        generation = solve_magic_square(square_size=magic_square_test_case.square_size, population_size=magic_square_test_case.population_size, num_generations=NUM_GENERATIONS, elite_percent=magic_square_test_case.elite_percent, mutation_rate=magic_square_test_case.mutation_rate, crossover_method=magic_square_test_case.crossover_method, validate_children=VALIDATE_CHILDREN)
        post_time = time.perf_counter_ns()

        # Check that the generations is less than the number of generations. If it isn't, then we didn't get a valid generation. Only append to the generation list when a valid value was returned from the solver.
        if generation < NUM_GENERATIONS:
            print("Solution attempt " + str(attempt_index + 1) + " solved the magic square in " + str(generation) + " generations")
            generations.append(generation)
        else:
            print("Solution attempt " + str(attempt_index + 1) + " could not find a solution for magic square in " + str(generation) + " generations")
            dnfs += 1

        print("Time to Process: " + str((post_time - pre_time) * 0.000001) + " ms")

    # Ensure generations has at least one value (the maximum, as if it is empty, then we got all DNFs) so fmean doesn't error.
    if len(generations) == 0:
        generations.append(NUM_GENERATIONS)

    # If we need to output to csv file, write the data line and append it to the csv data line list.
    if output_to_csv:
        csv_line = str(magic_square_test_case) + "," + str(int(statistics.fmean(generations)))

        if dnfs > 0:
            csv_line += " (" + str(dnfs) + " DNFs)"

        csv_lines.append(csv_line)

    if magic_square_test_case != magic_square_test_cases[-1]:
        print()

# If we want to output to csv, write to the output file the CSV header for every column, then write every csv line to file.
if output_to_csv:
    with open("output.csv", "w") as file:
        file.write("Crossover Method,Population Size,Mutation Rate,Elite Percent,Magic Square Size,Average Number of Generations Required (" + str(MAX_ATTEMPTS) + " runs)" + "\n")

        for csv_line in csv_lines:
            file.write(csv_line + "\n")
