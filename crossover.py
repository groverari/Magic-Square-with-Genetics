from enum import Enum, auto
import random

class CrossoverMethod(Enum):
    SINGLE_CROSSOVER_POINT = auto()
    SINGLE_CROSSOVER_POINT_PERCENTAGE = auto()

    SEQUENTIAL_SEGMENT_CROSSOVER = auto()
    SEQUENTIAL_SEGMENT_CROSSOVER_PERCENTAGE = auto()

    RANDOM_SEGMENT_CROSSOVER = auto()
    RANDOM_SEGMENT_CROSSOVER_PERCENTAGE = auto()

    ORDER_CROSSOVER = auto()

    VOTING_RECOMBINATION = auto()

    ALTERNATING_POSITION_CROSSOVER = auto()

def execute_crossover_method(parent1, parent2, crossover_method : CrossoverMethod):
    # Ensure that the lengths of the parents are equal.
    assert len(parent1) == len(parent2)

    # Python has implicit breaks, so flow-through isn't a problem here.
    match crossover_method:
        case CrossoverMethod.SINGLE_CROSSOVER_POINT:
            child1, child2 = single_point_crossover(parent1, parent2)
        case CrossoverMethod.SINGLE_CROSSOVER_POINT_PERCENTAGE:
            child1, child2 = single_point_crossover_percentage(parent1, parent2)

        case CrossoverMethod.SEQUENTIAL_SEGMENT_CROSSOVER:
            child1, child2 = sequential_segment_crossover(parent1, parent2)
        case CrossoverMethod.SEQUENTIAL_SEGMENT_CROSSOVER_PERCENTAGE:
            child1, child2 = sequential_segment_crossover_percentage(parent1, parent2)

        case CrossoverMethod.RANDOM_SEGMENT_CROSSOVER:
            child1, child2 = random_segment_crossover(parent1, parent2)
        case CrossoverMethod.RANDOM_SEGMENT_CROSSOVER_PERCENTAGE:
            child1, child2 = random_segment_crossover_percentage(parent1, parent2)

        case CrossoverMethod.ORDER_CROSSOVER:
            child1, child2 = order_crossover(parent1, parent2)

        case CrossoverMethod.VOTING_RECOMBINATION:
            child1, child2 = voting_recombination(parent1, parent2)

        case CrossoverMethod.ALTERNATING_POSITION_CROSSOVER:
            child1, child2 = alternating_position_crossover(parent1, parent2)

        case _:
            raise Exception("Unsupported CrossoverMethod passed to selection: " + str(crossover_method))
        
    assert len(child1) == len(child2)
        
    return child1, child2

# Will perform a crossover, produces two children which have taken from one parent a segment from two indices and the rest of the values (either randomly or sequentially) from the other parent.
def _single_segment_crossover(parent1, parent2, crossover_point_min = 4, crossover_point_max_length_offset = -4, randomize_segment = False):
    square_length = len(parent1)

    crossover_point_max = square_length + crossover_point_max_length_offset

    child1 = [0] * square_length
    child2 = [0] * square_length
    for child, child_parent1, child_parent2 in [(child1, parent1, parent2), (child2, parent2, parent1)]:
        # Copy values from crossover into child.
        for i in range(crossover_point_min, crossover_point_max + 1):
            child[i] = child_parent1[i]

        list = random.sample(child_parent2, len(child_parent2)) if randomize_segment else child_parent2

        # TODO: This is inefficient, as explicitly iterating the ranges which the crossover is not in will be faster than iterating all indices.
        # Append the rest of the values, checking that they are not already in the child.
        for i in range(square_length):
            if child[i] == 0:
                for value in list:
                    if value not in child:
                        child[i] = value
                        break

    return child1, child2

def random_segment_crossover(parent1, parent2, crossover_point_min = 4, crossover_point_max_length_offset = -4):
    return _single_segment_crossover(parent1=parent1, parent2=parent2, crossover_point_min=crossover_point_min, crossover_point_max_length_offset=crossover_point_max_length_offset, randomize_segment=True)

def random_segment_crossover_percentage(parent1, parent2, crossover_point_min_percentage = 0.25, crossover_point_max_percentage = 0.75):
    square_length = len(parent1)

    crossover_point_min = int(square_length * crossover_point_min_percentage)
    crossover_point_max = int(square_length * crossover_point_max_percentage)

    return random_segment_crossover(parent1=parent1, parent2=parent2, crossover_point_min=crossover_point_min, crossover_point_max_length_offset=crossover_point_max - square_length)

# Taken From 05/18/2023 11:50 AM: https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Crossover_for_permutations
def sequential_segment_crossover(parent1, parent2, crossover_point_min = 4, crossover_point_max_length_offset = -4):
    return _single_segment_crossover(parent1=parent1, parent2=parent2, crossover_point_min=crossover_point_min, crossover_point_max_length_offset=crossover_point_max_length_offset, randomize_segment=False)

def sequential_segment_crossover_percentage(parent1, parent2, crossover_point_min_percentage = 0.25, crossover_point_max_percentage = 0.75):
    square_length = len(parent1)

    crossover_point_min = int(square_length * crossover_point_min_percentage)
    crossover_point_max = int(square_length * crossover_point_max_percentage)

    return sequential_segment_crossover(parent1=parent1, parent2=parent2, crossover_point_min=crossover_point_min, crossover_point_max_length_offset=crossover_point_max - square_length)

# Perform single-point crossover between two parents
# swaps numbers between two parents
def single_point_crossover(parent1, parent2, crossover_point_min = 1, crossover_point_max_length_offset = -2):
    crossover_point = random.randint(crossover_point_min, len(parent1) + crossover_point_max_length_offset)
    child1 = parent1[:crossover_point] + \
        [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + \
        [gene for gene in parent1 if gene not in parent2[:crossover_point]]
    return child1, child2

def single_point_crossover_percentage(parent1, parent2, crossover_point_min_percentage = 0.25, crossover_point_max_percentage = 0.75):
    square_length = len(parent1)

    crossover_point_min = int(square_length * crossover_point_min_percentage)
    crossover_point_max = int(square_length * crossover_point_max_percentage)

    return single_point_crossover(parent1=parent1, parent2=parent2, crossover_point_min=crossover_point_min, crossover_point_max_length_offset=crossover_point_max - square_length)

def order_crossover(parent1, parent2, segment_amount = 10, maximum_segment_length = 10, maximum_segment_generation_attempts = 1):
    square_length = len(parent1)

    child1 = [0] * square_length
    child2 = [0] * square_length
    for child, child_parent1, child_parent2 in [(child1, parent1, parent2), (child2, parent2, parent1)]:
        for _ in range(segment_amount):
            # TODO: Relatively inefficient and no-guarantees way to do this.
            for _ in range(maximum_segment_generation_attempts):
                segment_start_index = random.randint(0, square_length - 1)
                segment_end_index = segment_start_index

                # Check if index is a valid starting point for a segment.
                if child[segment_start_index] == 0:
                    # Attempt to shift start and end indices randomly up to the valid length.
                    for _ in range(maximum_segment_length - 1):
                        # If the random number is 0, attempt to shift the start index to the left, else try to shift the end index to the right.
                        if random.randint(0, 1) == 0:
                            new_start_index = segment_start_index - 1
                            if new_start_index >= 0 and child[new_start_index] == 0:
                                segment_start_index = new_start_index
                        else:
                            new_end_index = segment_end_index + 1
                            if new_end_index < square_length and child[new_end_index] == 0:
                                segment_end_index = new_end_index

                    # At this point, have valid segment, so copy from child_parent1.
                    for i in range(segment_start_index, segment_end_index + 1):
                        child[i] = child_parent1[i]

                    # Break, as we successfully copied a segment.
                    break

        # Finally, copy the values sequentially from child_parent2 into the empty spots.
        for i in range(square_length):
            if child[i] == 0:
                for value in child_parent2:
                    if value not in child:
                        child[i] = value
                        break

    return child1, child2

# Taken From 05/18/2023 2:04 PM: https://www.researchgate.net/figure/13-Voting-recombination-example_fig19_29651423
def voting_recombination(parent1, parent2):
    square_length = len(parent1)

    child1 = [0] * square_length
    child2 = [0] * square_length

    for i in range(square_length):
        if parent1[i] == parent2[i]:
            child1[i] = parent1[i]
            child2[i] = parent1[i]

    for child, child_parent1 in [(child1, parent2), (child2, parent1)]:
        random_parent = random.sample(child_parent1, square_length)
        for i in range(square_length):
            if child[i] == 0:
                for value in random_parent:
                    if value not in child:
                        child[i] = value
                        break

    return child1, child2

# Taken From 05/18/2023 2:04 PM: https://www.researchgate.net/figure/Alternating-position-crossover-AP_fig5_226665831
def alternating_position_crossover(parent1, parent2):
    square_length = len(parent1)

    child1 = [0] * square_length
    child2 = [0] * square_length

    for child, child_parent1, child_parent2 in [(child1, parent1, parent2), (child2, parent2, parent1)]:
        parent1_index = 0
        parent2_index = 1

        for i in range(square_length):
            for _ in range(square_length):
                added_number = False

                if parent1_index < parent2_index:
                    parent1_value = child_parent1[parent1_index]
                    if parent1_value not in child:
                        child[i] = parent1_value
                        added_number = True

                    parent1_index += 1
                else:
                    parent2_value = child_parent2[parent2_index]
                    if parent2_value not in child:
                        child[i] = parent2_value
                        added_number = True
                    
                    parent2_index += 1

                if added_number:
                    break

    return child1, child2