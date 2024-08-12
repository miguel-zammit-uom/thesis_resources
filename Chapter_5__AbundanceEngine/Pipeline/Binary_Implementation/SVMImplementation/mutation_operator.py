import numpy as np


def _mutation(chromosome, accepted_ranges):
    """Those Pareto fronts wihich are not first but were not selected for replacement will be mutated.
       The number of these chromosomes is defined approximately half of the chromosomes where are not in the first
       Pareto front.

        For each chromosome:
            1. Randomly select how many genes will be mutated.
            2. Randomly select which genes will be mutated.
            3. A new random value is generated within the accepted range.
    """

    mutated_genes_num = np.random.randint(0, len(chromosome))
    mutated_genes_index = sorted(np.random.choice(range(len(chromosome)), size=mutated_genes_num, replace=False))

    for i in mutated_genes_index:
        gene = chromosome[i]
        new_gene = None

        # Checking if the gene is a cluster centre coordinate
        if i < len(chromosome)-6:
            new_gene = gene + np.random.uniform(low=-1, high=1)*gene
        else: # One of the final six hyperparameters
            if i == len(chromosome) - 6:    # fuzziness
                new_gene = np.random.uniform(low=accepted_ranges['fuzz_min'], high=accepted_ranges['fuzz_max'])
            elif i == len(chromosome) - 5:  # flexibility
                new_gene = np.random.uniform(low=accepted_ranges['flex_min'], high=accepted_ranges['flex_max'])
            elif i == len(chromosome) - 4:  # Kernel Function
                new_gene = np.random.randint(low=1, high=5)
            elif i == len(chromosome) - 3:  # gamma
                new_gene = np.random.uniform(low=accepted_ranges['gamma_min'], high=accepted_ranges['gamma_max'])
            elif i == len(chromosome) - 2:  # Degree
                new_gene = np.random.randint(low=accepted_ranges['d_min'], high=(accepted_ranges['d_max'] + 1))
            elif i == len(chromosome) - 1:  # r
                new_gene = np.random.uniform(low=accepted_ranges['r_min'], high=accepted_ranges['r_max'])

        chromosome[i] = new_gene

    return chromosome


class MutationOperator:
    """Those Pareto fronts which are not first but were not selected for replacement will be mutated.
    The number of these chromosomes is defined approximately half of the chromosomes where are not in the first Pareto
    front.

    For each chromosome:
        1. Randomly select how many genes will be mutated.
        2. Randomly select which genes will be mutated.
        3. A new random value is generated within the accepted range.
    """

    def __init__(self, population, i_chromosomes_for_mutation, accepted_ranges):

        self.mutated_chromosomes = list()

        for chromosome_index in i_chromosomes_for_mutation:
            self.mutated_chromosomes.append(_mutation(chromosome=population[chromosome_index],
                                                      accepted_ranges=accepted_ranges))

