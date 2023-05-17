from magic_square import solve_magic_square

generations = {}
for x in range(2, 10):
    for y in range(5):
        generations[x] = []
        generation = solve_magic_square(x+1)
        print(generation)
        generations[x].append(generation)
