import numpy as np


def _crossover_operation(parent_chromosomes):

    parent1, parent2 = parent_chromosomes
    child_chromosome_len = len(parent_chromosomes[np.random.randint(0, 2)])

    random_vector_len = min([len(parent1), len(parent2)])
    random_vector = np.random.randint(0, 2, random_vector_len)

    parent1_centres, parent1_fixed = parent1[:-9], parent1[-9:]
    parent2_centres, parent2_fixed = parent2[:-9], parent2[-9:]
    random_vector_centres, random_vector_fixed = random_vector[:-9], random_vector[-9:]

    child_chromosome = []
    for i in range(child_chromosome_len):
        if i < child_chromosome_len-9:     # The cluster centres genes
            if i < random_vector_len-9:    # The centre genes for which a random number in the vector was generated
                if random_vector_centres[i] == 0:
                    child_chromosome.append(parent1_centres[i])
                elif random_vector_centres[i] == 1:
                    child_chromosome.append(parent2_centres[i])
                else:
                    print('ERROR: Crossover Operator For the Following Chromosomes has Failed')
                    print(parent1, parent2)
            else:
                if len(parent1) > len(parent2):
                    child_chromosome.append(parent1[i])
                else:
                    child_chromosome.append(parent2[i])
        else:   # The final 9 genes
            if random_vector_fixed[i-child_chromosome_len] == 0:
                child_chromosome.append(parent1_fixed[i-child_chromosome_len])
            elif random_vector_fixed[i-child_chromosome_len] == 1:
                child_chromosome.append(parent2_fixed[i-child_chromosome_len])
            else:
                print('ERROR: Crossover Operator For the Following Chromosomes has Failed')
                print(parent1, parent2)

    return child_chromosome


def _crossover_selection(population, fitnesses, C_r):

    # Population - Fitnesses Error Sanity Check
    if len(population) == len(fitnesses):

        # Randomly select C_r chromosomes for crossover population
        selected_crossover_indices = np.random.choice(range(len(population)), size=C_r, replace=False)
        selected_crossover_pop = population[selected_crossover_indices]
        selected_crossover_fitnesses = fitnesses[selected_crossover_indices]

        # Select Best Performer (Max Value) for Obj 1 and Obj 2 (Mean value for all 3 clf fitnesses)
        best_performer_obj1_index = np.argmax(selected_crossover_fitnesses[:, 0])
        best_performer_obj2_index = np.argmax(np.mean(selected_crossover_fitnesses[:, 1:], axis=1))

        best_performer_obj1, best_performer_obj2 = selected_crossover_pop[best_performer_obj1_index], \
                                                   selected_crossover_pop[best_performer_obj2_index]

        # Select Worst Performer (Min Value) for Obj 1 and Obj 2 (Mean value for all 3 clf fitnesses)
        worst_performer_obj1_index = np.argmin(selected_crossover_fitnesses[:, 0])
        worst_performer_obj2_index = np.argmin(np.mean(selected_crossover_fitnesses[:, 1:], axis=1))

        worst_performer_obj1, worst_performer_obj2 = selected_crossover_pop[worst_performer_obj1_index], \
                                                     selected_crossover_pop[worst_performer_obj2_index]

        # The function returns the 3 sets of parent configurations which will be used to generate offspring
        return [best_performer_obj1, best_performer_obj2], \
               [best_performer_obj1, worst_performer_obj2], \
               [worst_performer_obj1, best_performer_obj2]

    else:
        print('RUNTIME ERROR: population array length is not equal to fitness array length')


class FullCrossoverOperator:
    def __init__(self, population, fitnesses, C_r):
        """
            Select C_r number of chromosomes randomly from all chromosomes in the population. Choose the 2 best
            performers (best for obj 1 and best for obj 2) and 2 worst (worst for obj 1 and worst for obj 2). These 4
            will be parents of 3 offspring

            1 set of Crossover operations generates 3 offspring:
                Offspring 1: Best obj 1 with Best obj 2
                Offspring 2: Best obj 1 with Worst obj 2
                Offspring 3: Best obj 2 with Worst obj 1

            For the crossover itself the following steps are used:
                1. Randomly select one of the parents to be the same size of the offspring.
                2. A random vector is generated with binary values. The length is equal to the minimum length of parents
                3. Crossover happens dictated by random vector.
        """

        self.population = population
        self.fitnesses = fitnesses

        self.parent_configurations = _crossover_selection(population=self.population, fitnesses=self.fitnesses,
                                                          C_r=C_r)
        self.generated_offspring = []
        for parents in self.parent_configurations:
            self.generated_offspring.append(_crossover_operation(parents))
