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
        if i < len(chromosome)-9:
            new_gene = gene + np.random.uniform(low=-1, high=1)*gene
        else: # One of the final six hyperparameters
            if i == len(chromosome) - 9:    # fuzziness
                new_gene = np.random.uniform(low=accepted_ranges['fuzz_min'], high=accepted_ranges['fuzz_max'])
            elif i == len(chromosome) - 8:  # n_estimators
                new_gene = np.random.randint(low=accepted_ranges["n_est_min"], high=accepted_ranges["n_est_max"])
            elif i == len(chromosome) - 7:  # max_depth
                new_gene = np.random.randint(low=accepted_ranges["max_depth_min"], high=accepted_ranges["max_depth_max"]
                                             )
            elif i == len(chromosome) - 6:  # learning_rate
                new_gene = np.random.choice([10**i for i in range(-accepted_ranges["lr_smallest_order"], 1)])
            elif i == len(chromosome) - 5:  # subsample
                new_gene = np.random.uniform(low=accepted_ranges["subsample_min"], high=accepted_ranges["subsample_max"]
                                             )
            elif i == len(chromosome) - 4:  # col_sample
                new_gene = np.random.uniform(low=accepted_ranges["col_sample_min"],
                                             high=accepted_ranges["col_sample_max"])
            elif i == len(chromosome) - 3:  # gamma
                new_gene = np.random.uniform(low=accepted_ranges['gamma_min'], high=accepted_ranges['gamma_max'])
            elif i == len(chromosome) - 2:  # reg_alpha
                new_gene = np.random.choice([10**i for i in range(-accepted_ranges["reg_alpha_smallest_order"], 1)])
            elif i == len(chromosome) - 1:  # reg_lambda
                new_gene = np.random.choice([10**i for i in range(-accepted_ranges["reg_lambda_smallest_order"], 1)])

        chromosome[i] = new_gene

    return chromosome


class MutationOperator:
    """Those Pareto fronts wihich are not first but were not selected for replacement will be mutated.
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

