import random
import sys

sys.setrecursionlimit(20000)
a, b, c = False, False, False
generation = 0
Np = 0
pop_size = 0
pop = []  # pop = Population List
fit = []  # fit = Fitness List
l, m, n, o = False, False, False, False
crossover_flag, crossover_choice, crossover_rate = False, 0, 0
p, r, mutation_rate, inversion_rate = 0, 0, 0, 0
tournament_size = 0
generation = 0
mutation_flag, mutation_choice = False, 0
mask = list()
prev_generation = -1


## Merge Sort


def merge_sort(pop, fit, n):
    if len(pop) <= 1:
        return

    mid = len(pop) // 2
    left_pop = pop[:mid]
    right_pop = pop[mid:]
    left_fit = fit[:mid]
    right_fit = fit[mid:]

    merge_sort(left_pop, left_fit, n)
    merge_sort(right_pop, right_fit, n)

    merge(pop, fit, left_pop, right_pop, left_fit, right_fit, n)


def merge(pop, fit, left_pop, right_pop, left_fit, right_fit, n):
    i = j = k = 0

    while i < len(left_pop) and j < len(right_pop):
        if n is True:
            if left_fit[i] <= right_fit[j]:
                pop[k] = left_pop[i]
                fit[k] = left_fit[i]
                i += 1
            else:
                pop[k] = right_pop[j]
                fit[k] = right_fit[j]
                j += 1
            k += 1
        if n is False:
            if left_fit[i] >= right_fit[j]:
                pop[k] = left_pop[i]
                fit[k] = left_fit[i]
                i += 1
            else:
                pop[k] = right_pop[j]
                fit[k] = right_fit[j]
                j += 1
            k += 1
    while i < len(left_pop):
        pop[k] = left_pop[i]
        fit[k] = left_fit[i]
        i += 1
        k += 1

    while j < len(right_pop):
        pop[k] = right_pop[j]
        fit[k] = right_fit[j]
        j += 1
        k += 1


## Fitness Calculation


def fitness(pop, string, fit):
    global generation, o
    fit = []
    best_string = 0
    for i in pop:
        ans = 0
        for j in range(len(string)):
            if (i[j] == string[j]):
                ans += 1
        current_fitness = ans / len(string)
        fit.append(format(current_fitness, ".3f"))
    max_value=0.0
    for i in range(0,len(fit)):
        if(max_value<float(fit[i])):
            max_value=float(fit[i])
            best_string=pop[i]
    print("\nNumber of Generations : ", generation, " Fitness : ",max_value,"best string",best_string)
    if (o == False and int(float(max_value)) == 1):
        print("\nConvergence Condition Met in 0th Generation\n")
        quit()
    o = True
    convergence(pop, fit)


## Convergence


def convergence(pop, fit):
    flag = 0
    global l, m, p, r, generation, loop, array
    for i in range(len(fit)):
        if int(float(fit[i])) == 1:
            print("\nConvergence Condition Met")
            print("\nString Length : ", len(string))
            print("\nLength of pop : ", pop_size)
            print("\nPercentage for Selection to Mating Pool : ", p, "%")
            print("\nSelection Method : ", S[int(r) - 1], "\nCrossover Method : ", C[int(crossover_choice) - 1],
                  "\nMutation Method  : ", M[int(mutation_choice) - 1])
            print("\nThe String is : ", pop[i])
            print("\nNumber of Generations Taken : ", generation, "\n")
            quit()
    if (l is False):
        p = int(input("\nEnter the Percentage for Selection to Mating Pool : "))
        l = True
        print(
            "\nSelection Methods\n1. Canonical Selection\n2. Rank-Based Selection\n3. Tornament Selection\n4. Roulette-Wheel Selection")
    if (m is False):
        r = input("\nChoose the Selection Method : ")
        m = True
    match r:
        case '1':
            generation += 1
            canonical_selection(pop, fit, p)
        case '2':
            generation += 1
            rank_based_selection(pop, fit, p)
        case '3':
            generation += 1
            tournament_selection(pop, fit, p)

        case '4':
            generation += 1
            roulette_wheel_selection(pop, fit, p)
        case _:
            print("\nChoose a Valid Selection Method\n")


# Canonical Selection


def canonical_selection(pop, fit, p):
    merge_sort(pop, fit, False)
    cano_fit = []
    cano_select = []
    cano_sum = sum(float(f) for f in fit)
    avg = cano_sum / len(fit)
    for f in fit:
        cano_fit.append(format(float(f) / avg, ".3f"))
    Np = p * len(pop) / 100
    if Np < 1:
        print("\nNo Population is Selected for Mating Pool")
        print("\nConvergence Condition Not Met")
        print("\nNumber of Generations Taken : ", generation, "\n")
        quit()
    for i in range(int(Np)):
        cano_select.append(pop[i])
    crossover(cano_select)


## Rank - Based Selection


def rank_based_selection(pop, fit, p):
    merge_sort(pop, fit, True)
    ranks = [fit.index(fitness) + 1 for fitness in fit]
    sum_ranks = sum(ranks)
    prob = [rank / sum_ranks for rank in ranks]
    Np = int(p * len(pop) / 100)
    if Np < 1:
        print("\nNo Population is Selected for Mating Pool")
        print("\nConvergence Condition Not Met")
        print("\nNumber of Generations Taken : ", generation, "\n")
        quit()
    selected_indices = []
    for _ in range(Np):
        rand = random.random()
        cumulative_prob = 0
        for i in range(len(pop)):
            cumulative_prob += prob[i]
            if cumulative_prob >= rand:
                selected_indices.append(i)
                break
    selected_population = [pop[i] for i in selected_indices]

    crossover(selected_population)


## Tournament Selection


def tournament_selection(pop, fit, p):
    global a, tournament_size
    Np = p * len(pop) // 100
    if Np < 1:
        print("\nNo Population is Selected for Mating Pool")
        print("\nConvergence Condition Not Met")
        print("\nNumber of Generations Taken: ", generation, "\n")
        quit()

    select = []
    if (a is False):
        print("Enter the tournamnet size between 2 and ", len(pop), " : ")
        tournament_size = int(input())
        a = True
    if (2 <= tournament_size <= len(pop)):
        for _ in range(Np):
            # tournament_size = 4
            tournament = random.sample(range(len(pop)), tournament_size)
            tournament_fitness = [fit[i] for i in tournament]
            winner_index = max(tournament, key=lambda i: fit[i])
            winner_fitness = fit[winner_index]
            winner_individual = pop[winner_index]
            select.append(winner_individual)

        crossover(select)
    else:
        print("\nTournament Size is not valid\n")
        quit()

    ## Roulette Wheel Selection


def roulette_wheel_selection(pop, fit, p):
    print("\nRoulette Wheel Selection")
    sum_fitness = sum(float(f) for f in fit)
    prob = [float(fitness) / sum_fitness for fitness in fit]
    Np = int(p * len(pop) / 100)
    if Np < 1:
        print("\nNo Population is Selected for Mating Pool")
        print("\nConvergence Condition Not Met")
        print("\nNumber of Generations Taken : ", generation, "\n")
        quit()
    selected_indices = []
    for _ in range(Np):
        rand = random.random()
        cumulative_prob = 0
        for i in range(len(pop)):
            cumulative_prob += prob[i]
            if cumulative_prob >= rand:
                selected_indices.append(i)
                break
    selected_population = [pop[i] for i in selected_indices]
    crossover(selected_population)


## Single Point Crossover


def single_point_crossover(parent1, parent2):
    length = len(parent1)
    crossover_point = random.randint(1, length - 1)
    parent1 = list(parent1)
    parent2 = list(parent2)
    for i in range(crossover_point, len(parent1)):
        parent1[i], parent2[i] = parent2[i], parent1[i]
    parent1 = ''.join(parent1)
    parent2 = ''.join(parent2)
    return [parent1, parent2]


## Two Point Crossover


def two_point_crossover(parent1, parent2):
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    offspring1 = [None] * len(parent1)
    offspring2 = [None] * len(parent1)
    offspring1[point1:point2] = parent1[point1:point2]
    offspring2[point1:point2] = parent2[point1:point2]
    for i in range(0, len(parent1)):
        if i < point1:
            offspring1[i], offspring2[i] = parent2[i], parent1[i]
        elif i >= point2:
            offspring1[i], offspring2[i] = parent2[i], parent1[i]
    offspring1 = ''.join(offspring1)
    offspring2 = ''.join(offspring2)
    return [offspring1, offspring2]


## Multipoint Crossover


def multipoint_crossover(parent1, parent2):
    if (len(parent1) == 2):
        return single_point_crossover(parent1, parent2)
    parent1 = list(parent1)
    parent2 = list(parent2)
    no_of_points = random.randint(2, len(parent1) - 1)
    points = sorted(random.sample(range(1, len(parent1)), no_of_points))
    for i in points:
        parent1, parent2 = temp_multipoint_crossover(parent1, parent2, i)
    parent1 = ''.join(parent1)
    parent2 = ''.join(parent2)
    return [parent1, parent2]


def temp_multipoint_crossover(parent1, parent2, point):
    parent1 = list(parent1)
    parent2 = list(parent2)
    for i in range(point, len(parent1)):
        parent1[i], parent2[i] = parent2[i], parent1[i]
    parent1 = ''.join(parent1)
    parent2 = ''.join(parent2)
    return [parent1, parent2]


## Uniform Crossover


def uniform_crossover(parent1, parent2):
    length = len(parent1)
    parent1 = list(parent1)
    parent2 = list(parent2)
    for i in range(len(parent1)):
        coin = random.randint(0, 1)
        if (coin == 0):
            parent1[i], parent2[i] = parent2[i], parent1[i]
    parent1 = ''.join(parent1)
    parent2 = ''.join(parent2)
    return [parent1, parent2]


## Uniform Crossover With Mask


def uniform_crossover_with_mask(parent1, parent2):
    global generation
    global prev_generation
    global mask
    length = len(parent1)
    parent1 = list(parent1)
    parent2 = list(parent2)
    if (generation != prev_generation):
        mask = [random.randint(0, 1) for _ in range(len(parent1))]
        prev_generation = generation
    for i in range(len(parent1)):
        if (mask[i] == 0):
            parent1[i], parent2[i] = parent2[i], parent1[i]
    parent1 = ''.join(parent1)
    parent2 = ''.join(parent2)
    return [parent1, parent2]


## Half Uniform Crossover


def half_uniform_crossover(parent1, parent2):
    parent1 = list(parent1)
    parent2 = list(parent2)
    for i in range(len(parent1)):
        if parent1[i] != parent2[i]:
            coin = random.randint(0, 1)
            if coin == 0:
                parent1[i], parent2[i] = parent2[i], parent1[i]
    parent1 = ''.join(parent1)
    parent2 = ''.join(parent2)
    return [parent1, parent2]


## Three Parent Crossover


def three_parent_crossover(parent1, parent2, parent3):
    length = len(parent1)
    offspring = [None] * len(parent1)
    for i in range(len(parent1)):
        if parent1[i] == parent2[i]:
            offspring[i] = parent1[i]
        else:
            offspring[i] = parent3[i]
    offspring = ''.join(offspring)
    return [offspring]


## Crossovers


def crossover(matingpool):
    global crossover_flag, crossover_choice, crossover_rate
    offsprings = list()
    if (len(matingpool) < 2):
        print("\nCrossover is Not Possible as Population in Mating Pool for Crossover is less than Required Size")
        print("\nConvergence Condition Not Met")
        print("\nNumber of Generations Taken : ", generation, "\n")
        quit()
    if (crossover_flag == False):
        print(
            "\nCrossover Methods\n1. Single point crossover\n2. Two point crossover\n3. Multi point crossover\n4. Uniform crossover\n5. Uniform crossover with mask\n6. Half uniform crossover\n7. Three parent crossover\n")
        crossover_choice = input("Choose the Crossover Method : ")
        crossover_flag = True
        crossover_rate = float(input("Enter Crossover Rate between 0.6 to 0.9:\n"))
    if (0.6 <= crossover_rate <= 0.9):
        match crossover_choice:
            case '1':
                for i in range(0, len(matingpool)):
                    parents = random.sample(matingpool, 2)
                    if (random.random() < crossover_rate):
                        offsprings.extend(single_point_crossover(parents[0], parents[1]))
                    else:
                        offsprings.extend(parents)
            case '2':
                for i in range(0, len(matingpool)):
                    parents = random.sample(matingpool, 2)
                    if (random.random() < crossover_rate):
                        offsprings.extend(two_point_crossover(parents[0], parents[1]))
                    else:
                        offsprings.extend(parents)
            case '3':
                for i in range(0, len(matingpool)):
                    parents = random.sample(matingpool, 2)
                    if (random.random() < crossover_rate):
                        offsprings.extend(multipoint_crossover(parents[0], parents[1]))
                    else:
                        offsprings.extend(parents)
            case '4':
                for i in range(0, len(matingpool)):
                    parents = random.sample(matingpool, 2)
                    if (random.random() < crossover_rate):
                        offsprings.extend(uniform_crossover(parents[0], parents[1]))
                    else:
                        offsprings.extend(parents)
            case '5':
                for i in range(0, len(matingpool)):
                    parents = random.sample(matingpool, 2)
                    if (random.random() < crossover_rate):
                        offsprings.extend(uniform_crossover_with_mask(parents[0], parents[1]))
                    else:
                        offsprings.extend(parents)
            case '6':
                for i in range(0, len(matingpool)):
                    parents = random.sample(matingpool, 2)
                    if (random.random() < crossover_rate):
                        offsprings.extend(half_uniform_crossover(parents[0], parents[1]))
                    else:
                        offsprings.extend(parents)

            case '7':
                if (len(matingpool) < 3):
                    print(
                        "\nCrossover is Not Possible as Population in Mating Pool for Crossover is less than Required Size")
                    print("\nConvergence Condition Not Met")
                    print("\nNumber of Generations Taken : ", generation, "\n")
                    quit()
                for i in range(0, len(matingpool)):
                    parents = random.sample(matingpool, 3)
                    if (random.random() < crossover_rate):
                        offsprings.extend(three_parent_crossover(parents[0], parents[1], parents[2]))
                    else:
                        offsprings.extend(parents[2])
            case _:
                print("\nPlease Choose a Valid Crossover Method\n")
                quit()
    else:
        print("Not a Valid Crossover Rate ")
        quit()
    pop = offsprings
    mutate(pop)


## Mutation


def mutate(pop):
    global mutation_flag, mutation_rate, mutation_choice, b, c, inversion_rate
    if mutation_flag is False:
        print("\nMutation Methods\n1. Mutation Flipping\n2. Mutation Interchanging\n3. Mutation Reversing")
        mutation_choice = input("\nChoose the Mutation Method : ")
        mutation_flag = True
    # mutation_rate = random.uniform(0.1 / len(pop), 1 / len(pop))
    if (b is False):
        mutation_rate = float(input("Enter the mutation rate between 0 and 1:"))
        b = True
    if (0 <= mutation_rate <= 1):
        match mutation_choice:
            case '1':
                pop = mutation_flipping(pop, mutation_rate)
            case '2':
                pop = mutation_interchanging(pop, mutation_rate)
            case '3':
                pop = mutation_reversing(pop, mutation_rate)
            case _:
                print("\nPlease Choose a Valid Mutation Method\n")
                quit()
    else:
        print("\nMutation rate is not valid\n")
        quit()
    if (c is False):
        inversion_rate = float(input("\nEnter the inversion rate between 0.001 and 0.1 : "))
        c = True
    if (0.001 <= inversion_rate <= 0.1):
        pop = inversion(pop, inversion_rate)
    else:
        print("\nThe inversion_rate is not valid\n")
        quit()
    fitness(pop, string, fit)


## Mutation Flipping


def mutation_flipping(pop, mutation_rate):
    mutated_individual = pop.copy()  # Create a copy of the individual

    for i in range(len(mutated_individual)):
        for j in range(len(mutated_individual[i])):
            if random.random() < mutation_rate:
                # Flip the bit at the j-th position
                mutated_individual[i] = mutated_individual[i][:j] + str(1 - int(mutated_individual[i][j])) + \
                                        mutated_individual[i][j + 1:]

    return mutated_individual


## Mutation Interchanging


def mutation_interchanging(pop, mutation_rate):
    mutated_population = []
    for individual in pop:
        if random.random() < mutation_rate:
            pos1, pos2 = random.sample(range(len(individual)), 2)
            individual = list(individual)
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
            individual = ''.join(individual)
        mutated_population.append(individual)
    return mutated_population


## Mutation Reversing


def mutation_reversing(pop, mutation_rate):
    mutated_population = []
    for individual in pop:
        if random.random() < mutation_rate:
            start_pos = random.randint(0, len(individual) - 2)
            end_pos = random.randint(start_pos + 1, len(individual) - 1)
            individual = individual[:start_pos] + individual[start_pos:end_pos + 1][::-1] + individual[end_pos + 1:]
        mutated_population.append(individual)
    return mutated_population


## Inversion


def inversion(pop, inversion_rate):
    inverted_population = []
    for individual in pop:
        if random.random() < inversion_rate:
            mutated_individual = individual.replace('0', '2').replace('1', '0').replace('2', '1')
        else:
            mutated_individual = individual
        inverted_population.append(mutated_individual)
    return inverted_population


## Main





def ml():
    global pop_size
    for i in string:
        if i != '0' and i != '1':
            print("Given String is not Binary\n")
            quit()
    if (len(string) == len_string):
        pop_size = int(input("\nEnter the size of the Population : "))
        if (pop_size > 2 ** len_string):
            print("The strings cannot be generated\n")
            quit()
        pop = []
        binary_strings = set()
        while len(binary_strings) < pop_size:
            binary_string = ''.join(random.choice('01') for _ in range(len_string))
            binary_strings.add(binary_string)
        pop = list(binary_strings)
        fitness(pop, string, fit)
    else:
        print("\nThe string size doesnot match with the given length!!\n")
        quit()


if __name__ == '__main__':
    len_string = int(input("\nEnter the length of the string: "))
    string = input("\nEnter the string: ")
    S = ['Canonical Selection', 'Rank-Based Selection', 'Tournament Selection', 'Roulette-Wheel Selection']
    C = ['Single Point Crossover', 'Two Point Crossover', 'Multi Point Crossover', 'Uniform Crossover',
         'Uniform Crossover with Mask', 'Half Uniform Crossover', 'Three Parent Crossover']
    M = ['Mutation Flipping', 'Mutation Interchanging', 'Mutation Reversing']
    ml()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
