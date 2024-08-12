import numpy as np
from config_reader import ConfigReader
from data_prep import DataPreparation
from population_management import InitialPopulationGeneration, _population_recombination, _final_pareto_fronts_storage, \
    _chromosome_history_storage
from imputation_and_fitness import DatasetImputation, FitnessCalculation
from tournament_selection import TournamentSelection
from mutation_operator import MutationOperator
from crossover_operator import FullCrossoverOperator
from stopping_criteria_check import StoppingCriteriaCheck
from tqdm import tqdm
from gen_alg_analytics import _upr_border_print, _lwr_border_print, _mid_border_print, _information_print

# Read Config
_upr_border_print()
print('Initiating Run and Loading Metadata...')
configreader = ConfigReader(configfile='config.ini')
_lwr_border_print()

# Load Data & Prepare
_upr_border_print()
print('Loading & Preparing Dataset...\n\tDataset Selected: ', configreader.dataset_name)
dataprep = DataPreparation(traindataset_path=configreader.dataset_path,
                           testdataset_path=configreader.dataset_path,
                           element_features_list=configreader.element_features_list)
x_train, x_test = np.copy(dataprep.x_train), np.copy(dataprep.x_test)
print('\tDataset Loaded.')
_lwr_border_print()

# Generate Initial Population
_upr_border_print()
print('Generating Initial Population...')
init_pop_gen = InitialPopulationGeneration(pop_size=configreader.pop_size, max_clusters=configreader.max_clusters,
                                           accepted_ranges=configreader.accepted_ranges)
population = init_pop_gen.population
print('Population Generated. Chromosomes are Partially Generated & Ready\n for Imputation and Fuzzy Clustering\n',
      '        Number of Chromosomes: ', population.shape[0])
_lwr_border_print()

prev_gen_avg_fitness, fitnesses, accuracy_jup_scores, accuracy_jup_std, f1_jup_scores = {}, [], [], [], []
f1_jup_std, accuracy_rocky_scores, accuracy_rocky_std, f1_rocky_scores, f1_rocky_std = [], [], [], [], []
accuracy_multi_scores, accuracy_multi_std, f1_multi_scores, f1_multi_std = [], [], [], []

gen_1_mean_imp_x_train, gen_1_mean_imp_x_test = np.array([]), ([])
final_pareto_front = []
for generation in range(1, configreader.max_generations + 1):
    """Iterate over successive generations"""
    _upr_border_print()
    print('Generation ', generation)
    _mid_border_print()

    # Calculate Fitnesses
    fitnesses, full_chromosome_population, accuracy_jup_scores, accuracy_jup_std, f1_jup_scores = [], [], [], [], []
    f1_jup_std, accuracy_rocky_scores, accuracy_rocky_std, f1_rocky_scores, f1_rocky_std = [], [], [], [], []
    accuracy_multi_scores, accuracy_multi_std, f1_multi_scores, f1_multi_std = [], [], [], []
    for chromosome in tqdm(population, desc='        Fitness Calculation'):

        dataset_imp = DatasetImputation(x_train=x_train, x_test=x_test,
                                        x_train_missing_i=dataprep.x_train_missing_i,
                                        x_test_missing_i=dataprep.x_test_missing_i, chromosome=chromosome,
                                        generation_number=generation, features_list=configreader.element_features_list,
                                        log_absolute_conversion=configreader.log_absolute_conversion)
        full_chromosome_population.append(dataset_imp.full_chromosome)

        # Update original datasets with latest imputed values
        x_train, x_test = np.copy(dataset_imp.x_train_imputed), np.copy(dataset_imp.x_test_imputed)

        # Cloning the mean_imputed datasets so that prior to the imputation for each chromosome in successive
        # generations, the membership calculation is done on the mean_imputed dataset.
        if generation == 1:
            gen_1_mean_imp_x_train, gen_1_mean_imp_x_test = np.copy(dataset_imp.x_train_imputed), \
                                                            np.copy(dataset_imp.x_test_imputed)

        fitness_calc = FitnessCalculation(chromosome=dataset_imp.full_chromosome,
                                          x_train_imputed=x_train, y_train=dataprep.y_train,
                                          x_test_imputed=x_test,
                                          x_train_cluster_labels=dataset_imp.x_train_cluster_labels,
                                          x_test_cluster_labels=dataset_imp.x_test_cluster_labels,
                                          cluster_val_fn_choice=configreader.cluster_validity_fn,
                                          cross_val=configreader.cross_val,
                                          num_datasubsamples=configreader.num_datasubsamples,
                                          label_error_run_name=configreader.run_name,
                                          clf_metric=configreader.clf_metric)

        fitnesses.append([fitness_calc.fitness_1, fitness_calc.fitness_2, fitness_calc.fitness_3,
                          fitness_calc.fitness_4])

        accuracy_jup_scores.append(fitness_calc.accuracy_jup[0])
        accuracy_jup_std.append(fitness_calc.accuracy_jup[1])
        f1_jup_scores.append(fitness_calc.f1_jup[0])
        f1_jup_std.append(fitness_calc.f1_jup[1])
        accuracy_rocky_scores.append(fitness_calc.accuracy_rocky[0])
        accuracy_rocky_std.append(fitness_calc.accuracy_rocky[1])
        f1_rocky_scores.append(fitness_calc.f1_rocky[0])
        f1_rocky_std.append(fitness_calc.f1_rocky[1])
        accuracy_multi_scores.append(fitness_calc.accuracy_multi[0])
        accuracy_multi_std.append(fitness_calc.accuracy_multi[1])
        f1_multi_scores.append(fitness_calc.f1_multi[0])
        f1_multi_std.append(fitness_calc.f1_multi[1])

        if generation > 1:
            # Set datasets back the original mean_imputed datasets from generation 1
            x_train, x_test = np.copy(gen_1_mean_imp_x_train), np.copy(gen_1_mean_imp_x_test)


    # Tournament Selection, Returns array of Pareto Fronts, Ultimately selecting the Surviving Chromosomes and those
    # to be replaced
    population, fitnesses, accuracy_jup_scores, accuracy_jup_std, f1_jup_scores, f1_jup_std \
        = np.array(full_chromosome_population), np.array(fitnesses), np.array(accuracy_jup_scores), \
          np.array(accuracy_jup_std), np.array(f1_jup_scores), np.array(f1_jup_std)
    accuracy_rocky_scores, accuracy_rocky_std, f1_rocky_scores, f1_rocky_std \
        = np.array(accuracy_rocky_scores), np.array(accuracy_rocky_std), \
          np.array(f1_rocky_scores), np.array(f1_rocky_std)
    accuracy_multi_scores, accuracy_multi_std, f1_multi_scores, f1_multi_std \
        = np.array(accuracy_multi_scores), np.array(accuracy_multi_std), \
          np.array(f1_multi_scores), np.array(f1_multi_std)
    tourn_selection = TournamentSelection(population=population, fitnesses=fitnesses,
                                          num_replacement_fronts=configreader.num_replacement_fronts)
    _mid_border_print()
    print("        Pareto Front : ", len(tourn_selection.first_pareto_front), " chromosomes")
    print("        Replacement Fronts: ", len(tourn_selection.replacement_fronts), " chromosomes")
    print("        Mutation Fronts: ", len(tourn_selection.mutation_fronts), " chromosomes")

    _chromosome_history_storage(population=population, fitnesses=fitnesses,
                                metrics=np.transpose(np.array([accuracy_jup_scores, accuracy_jup_std, f1_jup_scores,
                                                               f1_jup_std, accuracy_rocky_scores, accuracy_rocky_std,
                                                               f1_rocky_scores, f1_rocky_std, accuracy_multi_scores,
                                                               accuracy_multi_std, f1_multi_scores,
                                                               f1_multi_std])),
                                num_features=len(configreader.element_features_list),
                                generation=generation,
                                output_csv_filename=str('chromosome_histories/chromosome_history_'
                                                        + configreader.run_name + '.csv'))

    if generation == 1:
        avgs = [np.mean(fitnesses.transpose()[0][~np.isnan(fitnesses.transpose()[0])], axis=0),
                np.mean(fitnesses.transpose()[1][~np.isnan(fitnesses.transpose()[1])], axis=0)]
        maxs = [np.max(fitnesses.transpose()[0][~np.isnan(fitnesses.transpose()[0])], axis=0),
                np.max(fitnesses.transpose()[1][~np.isnan(fitnesses.transpose()[1])], axis=0)]
        acc_avg, acc_max = np.mean(accuracy_jup_scores[~np.isnan(accuracy_jup_scores)]), \
                           np.max(accuracy_jup_scores[~np.isnan(accuracy_jup_scores)])
        prev_gen_avg_fitness = {'Imputation': avgs[0],
                                'Classification': avgs[1]}

        _information_print(n_gen=generation, n_pareto=len(tourn_selection.first_pareto_front), avg_clf_metric=avgs[1],
                           max_clf_metric=maxs[1], avg_imp_score=avgs[0], max_imp_score=maxs[0], avg_clf_acc=acc_avg,
                           max_clf_acc=acc_max, tau_imp=None, tau_clf=None,
                           output_txt_filename=configreader.generations_stats_out_txt)
    else:
        avgs = [np.mean(fitnesses.transpose()[0][~np.isnan(fitnesses.transpose()[0])], axis=0),
                np.mean(fitnesses.transpose()[1][~np.isnan(fitnesses.transpose()[1])], axis=0)]
        maxs = [np.max(fitnesses.transpose()[0][~np.isnan(fitnesses.transpose()[0])], axis=0),
                np.max(fitnesses.transpose()[1][~np.isnan(fitnesses.transpose()[1])], axis=0)]
        acc_avg, acc_max = np.mean(accuracy_jup_scores[~np.isnan(accuracy_jup_scores)]), \
                           np.max(accuracy_jup_scores[~np.isnan(accuracy_jup_scores)])
        current_gen_avg_fitness = {'Imputation': avgs[0],
                                   'Classification': avgs[1]}
        tau_imp = current_gen_avg_fitness['Imputation'] - prev_gen_avg_fitness['Imputation']
        tau_clf = current_gen_avg_fitness['Classification'] - prev_gen_avg_fitness['Classification']

        _information_print(n_gen=generation, n_pareto=len(tourn_selection.first_pareto_front), avg_clf_metric=avgs[1],
                           max_clf_metric=maxs[1], avg_imp_score=avgs[0], max_imp_score=maxs[0], avg_clf_acc=acc_avg,
                           max_clf_acc=acc_max, tau_imp=tau_imp, tau_clf=tau_clf,
                           output_txt_filename=configreader.generations_stats_out_txt)

        # Check Stopping Criteria
        stopping_criteria = StoppingCriteriaCheck(first_pareto_front_size=len(tourn_selection.first_pareto_front),
                                                  current_gen_avg_fitness=prev_gen_avg_fitness,
                                                  prev_gen_avg_fitness=current_gen_avg_fitness,
                                                  max_pop_size=configreader.pop_size,
                                                  clf_thresh=configreader.clf_threshold,
                                                  imputation_thresh=configreader.imputation_threshold)
        prev_gen_avg_fitness = current_gen_avg_fitness
        if stopping_criteria.stop_run:
            final_pareto_front = tourn_selection.first_pareto_front.copy()
            print('Stopping Criteria Met: ', stopping_criteria.criteria_met)
            break

    # Mutation Operator
    _mid_border_print()
    print('        Mutating Chromosomes...')
    mutation_operator = MutationOperator(population=population,
                                         i_chromosomes_for_mutation=tourn_selection.mutation_fronts,
                                         accepted_ranges=configreader.accepted_ranges)

    # Crossover Operator
    print('        Generating Offspring Chromosomes...')
    _mid_border_print()
    offspring_chromosomes = []
    for i in range(int(np.ceil(len(tourn_selection.replacement_fronts) / 3))):
        crossover_operator = FullCrossoverOperator(population=population, fitnesses=fitnesses, C_r=configreader.c_r)
        offspring_chromosomes += crossover_operator.generated_offspring
    offspring_chromosomes = offspring_chromosomes[:len(tourn_selection.replacement_fronts)]
    _mid_border_print()

    population = _population_recombination(population=population, first_pareto_front=tourn_selection.first_pareto_front,
                                           mutated_chromosomes=mutation_operator.mutated_chromosomes,
                                           offspring_chromosomes=offspring_chromosomes)
    if generation == configreader.max_generations:
        final_pareto_front = tourn_selection.first_pareto_front.copy()
    del tourn_selection.first_pareto_front, tourn_selection.replacement_fronts, tourn_selection.mutation_fronts

    print('New Population Generated.')
    print('        Number of Chromosomes: ', len(population))
    _lwr_border_print()


_final_pareto_fronts_storage(population=population, fitnesses=fitnesses,
                             metrics=np.transpose(np.array([accuracy_jup_scores, f1_jup_scores,
                                                            accuracy_rocky_scores, f1_rocky_scores,
                                                            accuracy_multi_scores, f1_multi_scores,
                                                            ])),
                             first_pareto_front=final_pareto_front,
                             output_csv_filename=str('final_pareto_fronts/test_' + configreader.run_name + '.csv'))
