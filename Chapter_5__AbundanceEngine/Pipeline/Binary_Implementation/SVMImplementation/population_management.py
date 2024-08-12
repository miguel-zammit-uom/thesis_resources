import numpy as np


class InitialPopulationGeneration:

    def __init__(self, pop_size, max_clusters, accepted_ranges):
        """The inputs for this should be the relevant hyperparameters set for model. Randomly generate the first
        generation of chromosomes to start off the Genetic Algorithm. Note that the chromosomes generated here won't be
        'full', as the cluster centres will be determined after the first imputation of the missing values as described
        in DatasetImputation (imputation_and_fitness.py). At this stage for the imputation parameters, the chromosomes
        will have the number of clusters and the fuziness parameter."""

        self.pop_size = pop_size
        self.max_clusters= max_clusters
        self.accepted_ranges = accepted_ranges

        self.population = list()

        for i in range(self.pop_size):
            self._chromosome_generator()

        self.population = np.array(self.population)

    def _chromosome_generator(self):
        num_clusters = np.random.randint(low=2, high=(self.max_clusters+1))
        fuzziness = np.random.uniform(low=self.accepted_ranges['fuzz_min'], high=self.accepted_ranges['fuzz_max'])

        C = np.random.uniform(low=self.accepted_ranges['flex_min'], high=self.accepted_ranges['flex_max'])
        k_r = np.random.randint(low=1, high=5)
        gamma = np.random.uniform(low=self.accepted_ranges['gamma_min'], high=self.accepted_ranges['gamma_max'])
        d = np.random.uniform(low=self.accepted_ranges['d_min'], high=self.accepted_ranges['d_max'])
        r = np.random.uniform(low=self.accepted_ranges['r_min'], high=self.accepted_ranges['r_max'])

        self.population.append([num_clusters, fuzziness, C, k_r, gamma, d, r])


def _population_recombination(population, first_pareto_front, mutated_chromosomes, offspring_chromosomes):
    """New Population is Recombined from first pareto front, mutated chromosomes and new offspring.
       Make sure all 3 are lists. """

    pareto_chromosomes = [population[i] for i in first_pareto_front]

    new_population = pareto_chromosomes + mutated_chromosomes + offspring_chromosomes

    return new_population


def _chromosome_history_storage(population, fitnesses, metrics, generation, num_features, output_csv_filename):
    if generation == 1:
        csv_file = open(output_csv_filename, "w")
        csv_file.write('num_clusters,centres,fuzziness,C,K_r,gamma,d,r,Imputation_Fitness,Clf_Fitness,Clf_Accuracy,'
                       'Clf_Accuracy_err,Clf_f1,Clf_f1_err,Clf_Recall,Clf_Recall_err,Clf_Precision,Clf_Precision_err\n')
    else:
        csv_file = open(output_csv_filename, "a")

    for chromosome, fitness_vals, metrics_i in zip(population, fitnesses, metrics):
        centres = chromosome[:-6]
        num_clusters = int(len(centres) / num_features)
        fuzziness, C, K_r, gamma, d, r = chromosome[-6:]

        csv_file.write(str(num_clusters) + ',' + ';'.join(str(elt) for elt in centres) + ',' + str(fuzziness) + ','
                       + str(C) + ',' + str(K_r) + ',' + str(gamma) + ',' + str(d) + ',' + str(r)
                       + ',' + str(fitness_vals[0]) + ',' + str(fitness_vals[1]) + ',' + str(metrics_i[0]) + ','
                       + str(metrics_i[1]) + ',' + str(metrics_i[2]) + ',' + str(metrics_i[3]) + ',' + str(metrics_i[4]) + ','
                       + str(metrics_i[5]) + ',' + str(metrics_i[6]) + ',' + str(metrics_i[7])
                       )
        csv_file.write('\n')

    csv_file.close()


def _final_pareto_fronts_storage(population, fitnesses, metrics, first_pareto_front, output_csv_filename):
    """Save the data for the set of non-dominated solutions after the GA is stopped"""

    pareto_chromosomes = [population[i] for i in first_pareto_front]
    pareto_fitnesses = [fitnesses[j] for j in first_pareto_front]
    pareto_metrics = [metrics[k] for k in first_pareto_front]

    csv_file = open(output_csv_filename, "w")
    csv_file.write('Chromosome,Imputation_Fitness,Clf_Fitness,Clf_Accuracy,Clf_f1,Clf_Recall,Clf_Precision\n')
    for chromosome, fitness_vals, metrics_vals in zip(pareto_chromosomes, pareto_fitnesses, pareto_metrics):
        csv_file.write(';'.join(str(elt) for elt in chromosome) + ',' + str(fitness_vals[0]) + ',' +
                       str(fitness_vals[1]) + ',' + str(metrics_vals[0]) + ',' + str(metrics_vals[1]) + ','
                       + str(metrics_vals[2]) + ',' + str(metrics_vals[3]))
        csv_file.write('\n')
    csv_file.close()
