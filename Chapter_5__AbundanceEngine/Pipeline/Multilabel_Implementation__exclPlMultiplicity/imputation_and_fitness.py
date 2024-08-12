import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.base import clone
from joblib import Parallel, delayed
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.model_selection import iterative_train_test_split
import skfuzzy.cluster as fuzz
import xgboost as xgb


def _log_to_absolute_feature_conversion(dataset, features_list):
    non_log_features = 'f_ra,f_dec,f_x,f_y,f_z,f_dist,f_disk,f_spec,f_vmag,f_bv,f_u,f_v,f_w,f_teff,f_mass,f_radius,' \
                       'f_name,f_p,f_m_p,f_e,f_a,f_r_p,f_min_m_p,Jupiter_Host,Rocky_Host'.split(',')
    dataset_copy = np.copy(dataset)

    if len(features_list) != dataset.shape[1]:
        sys.exit("RUNTIME ERROR: Element Feature List does not correspond to number of features in data.")

    for i in range(dataset_copy.shape[0]):
        for j, feature in zip(range(len(features_list)), features_list):
            if feature not in non_log_features:
                if not np.isnan(dataset_copy[i, j]):
                    dataset_copy[i, j] = 10**dataset_copy[i, j]

    return dataset_copy


def _absolute_to_log_feature_conversion(dataset, features_list):
    non_log_features = 'f_ra,f_dec,f_x,f_y,f_z,f_dist,f_disk,f_spec,f_vmag,f_bv,f_u,f_v,f_w,f_teff,f_mass,f_radius,' \
                       'f_name,f_p,f_m_p,f_e,f_a,f_r_p,f_min_m_p,Jupiter_Host,Rocky_Host'.split(',')
    dataset_copy = np.copy(dataset)

    if len(features_list) != dataset.shape[1]:
        sys.exit("RUNTIME ERROR: Element Feature List does not correspond to number of features in data.")

    for i in range(dataset_copy.shape[0]):
        for j, feature in zip(range(0, len(features_list)), features_list):
            if feature not in non_log_features:
                if not np.isnan(dataset_copy[i, j]):
                    try:
                        dataset_copy[i, j] = np.log10(dataset_copy[i, j])
                    except RuntimeWarning:
                        print(f'Value: {dataset_copy[i, j]} | Feature: {feature}')

    return dataset_copy


def _dataset_subsampling(x_data, y_data, num_subsamples, balancesplit=0.6):
    """Subsamples the 0 label subset to decrease the imbalance of the dataset. balancesplit sets the fraction of
    the dataset with a 0 label. """

    x_data = np.delete(x_data, np.s_[-y_data.shape[1]:], axis=1)

    hosts_index = set(np.unique(np.argwhere(y_data != 0)[:, 0]))
    mask = np.array([(i in hosts_index) for i in range(len(y_data))])

    x_data_hosts, y_data_hosts = x_data[mask][:], y_data[mask][:]
    x_data_comparisons, y_data_comparisons = x_data[~mask][:], y_data[~mask][:]

    num_comparisons = int((len(y_data_hosts)*balancesplit)/(1-balancesplit))

    data_subsamples = []

    for i in range(num_subsamples):
        subsample_indices = np.random.choice(a=len(y_data_comparisons), size=num_comparisons, replace=False)
        x_data_comparisons_subsample, y_data_comparisons_subsample \
            = x_data_comparisons[subsample_indices], y_data_comparisons[subsample_indices]
        x_data_sample, y_data_sample = \
            np.concatenate((x_data_hosts, x_data_comparisons_subsample), axis=0), \
            np.concatenate((y_data_hosts, y_data_comparisons_subsample), axis=0)
        data_subsamples.append([x_data_sample.reshape((x_data_sample.shape[0], x_data_sample.shape[-1])),
                                y_data_sample.reshape((y_data_sample.shape[0], y_data_sample.shape[1]))])
        del subsample_indices, x_data_comparisons_subsample, y_data_comparisons_subsample, x_data_sample, y_data_sample

    return data_subsamples


def membership_calculation(datapoint, centres, fuzziness):
    num_features = len(datapoint)
    memberships = np.zeros(len(centres))
    distances_from_centres = []
    for i, centre in enumerate(centres):
        if centre.shape[0] != num_features:
            sys.exit("ERROR")
        distance = np.linalg.norm(datapoint - centre)
        distances_from_centres.append(distance)
        if distance == 0:
            memberships[i] = 1
            return memberships

    for j in range(len(memberships)):
        summation = np.zeros(len(memberships))
        for k in range(len(memberships)):
            summation[k] = np.power((distances_from_centres[j]**2)/(distances_from_centres[k]**2), (1/(fuzziness-1)))
        memberships[j] = 1/np.sum(summation)

    return memberships


def _mean_imputation(dataset, log_absolute_conversion, features_list):
    features_std_dev = []

    if log_absolute_conversion:
        # Convert Logarithmic values to Absolute Values
        dataset = _log_to_absolute_feature_conversion(dataset, features_list=features_list)

    for j, feature in enumerate(features_list):
        measured_values_index = np.argwhere(np.logical_not(np.isnan(dataset[:, j])))
        mean_value, std_dev = np.mean(dataset[measured_values_index, j]), np.std(dataset[measured_values_index, j])
        missing_values_index = np.argwhere(np.isnan(dataset[:, j]))

        if feature == "f_disk":
            dataset[missing_values_index, j] = np.round(mean_value)
        else:
            dataset[missing_values_index, j] = mean_value

        non_log_features = 'f_ra,f_dec,f_x,f_y,f_z,f_dist,f_disk,f_spec,f_vmag,f_bv,f_u,f_v,f_w,f_teff,f_mass,f_radiu' \
                           's,f_name,f_p,f_m_p,f_e,f_a,f_r_p,f_min_m_p,Jupiter_Host,Rocky_Host'.split(',')
        if log_absolute_conversion:
            if feature not in non_log_features:
                features_std_dev.append(np.std(np.log10(dataset[measured_values_index, j])))
            else:
                features_std_dev.append(std_dev)
        else:
            features_std_dev.append(std_dev)

    if log_absolute_conversion:
        # Convert Absolute values back to Logarithmic Values
        dataset = _absolute_to_log_feature_conversion(dataset, features_list)

    return dataset, np.array(features_std_dev)


def membership_imputation(feature_index, memberships, centres):
    summation = 0
    for i in range(len(centres)):
        summation += memberships[i]*centres[i, feature_index]
    return summation


def _average_silhouette_width(x_train_imputed, x_test_imputed, x_train_cluster_labels, x_test_cluster_labels,
                              chromosome, label_error_run_name):
    """Calculate Average Silhouette Width"""
    x_data = np.concatenate([x_train_imputed, x_test_imputed])
    labels = np.concatenate([x_train_cluster_labels, x_test_cluster_labels])
    if len(x_data) != len(labels):
        sys.exit("RUNTIME ERROR: Number of instances in concatenated dataset and number of labels does not "
                 "match")

    if len(np.unique(labels)) == 1:
        label_error_file = open(str('label_error_files/label_error_file'+label_error_run_name+'.txt'), 'a')
        label_error_file.write('RUNTIME ERROR: Number of unique cluster labels in concatenated dataset is 1. '
                               'ASW metric will be set to -1 to kill off chromosome\n\n\n')
        label_error_file.write('Affected Chromosome: \n')
        for gene in chromosome:
            label_error_file.write(str(gene) + ', ')
        label_error_file.write('\n\nUnique Cluster Labels: ')
        for uniquelabel in np.unique(labels):
            label_error_file.write(str(uniquelabel) + ', ')
        label_error_file.write('\n------------------------------------------------------------------\n\n')
        label_error_file.close()
        return np.nan
    elif np.isnan(x_data).any():
        imp_error_file = open(str('imputation_error_files/nan_imputation_error_file' + label_error_run_name + '.txt'),
                              'a')
        imp_error_file.write('RUNTIME ERROR: Imputation Error as some values in training set are nan '
                             'ASW metric will be set to -1 to kill off chromosome\n\n\n')
        imp_error_file.write('Affected Chromosome: \n')
        for gene in chromosome:
            imp_error_file.write(str(gene) + ', ')
        imp_error_file.write('\n------------------------------------------------------------------\n\n')
        imp_error_file.close()
        return np.nan
    elif np.isnan(labels).any():
        imp_error_file = open(str('imputation_error_files/nan_imputation_error_file' + label_error_run_name + '.txt'),
                              'a')
        imp_error_file.write('RUNTIME ERROR: Imputation Error as some labels are nan '
                             'ASW metric will be set to -1 to kill off chromosome\n\n\n')
        imp_error_file.write('Affected Chromosome: \n')
        for gene in chromosome:
            imp_error_file.write(str(gene) + ', ')
        imp_error_file.write('\n------------------------------------------------------------------\n\n')
        imp_error_file.close()
        return np.nan
    else:
        return silhouette_score(X=x_data, labels=labels)


def _davies_bouldin_score_calculation(x_train_imputed, x_test_imputed, x_train_cluster_labels, x_test_cluster_labels,
                                      chromosome, label_error_run_name):
    """Calculate DBI score"""
    x_data = np.concatenate([x_train_imputed, x_test_imputed])
    labels = np.concatenate([x_train_cluster_labels, x_test_cluster_labels])
    if len(x_data) != len(labels):
        sys.exit("RUNTIME ERROR: Number of instances in concatenated dataset and number of labels does not "
                 "match")

    if len(np.unique(labels)) == 1:
        label_error_file = open(str('label_error_files/label_error_file' + label_error_run_name + '.txt'), 'a')
        label_error_file.write('RUNTIME ERROR: Number of unique cluster labels in concatenated dataset is 1. '
                               'ASW metric will be set to -1 to kill off chromosome\n\n\n')
        label_error_file.write('Affected Chromosome: \n')
        for gene in chromosome:
            label_error_file.write(str(gene) + ', ')
        label_error_file.write('\n\nUnique Cluster Labels: ')
        for uniquelabel in np.unique(labels):
            label_error_file.write(str(uniquelabel) + ', ')
        label_error_file.write('\n------------------------------------------------------------------\n\n')
        label_error_file.close()
        return np.nan
    elif np.isnan(x_data).any():
        imp_error_file = open(str('imputation_error_files/nan_imputation_error_file' + label_error_run_name + '.txt'),
                              'a')
        imp_error_file.write('RUNTIME ERROR: Imputation Error as some values in training set are nan '
                             'ASW metric will be set to -1 to kill off chromosome\n\n\n')
        imp_error_file.write('Affected Chromosome: \n')
        for gene in chromosome:
            imp_error_file.write(str(gene) + ', ')
        imp_error_file.write('\n------------------------------------------------------------------\n\n')
        imp_error_file.close()
        return np.nan
    elif np.isnan(labels).any():
        imp_error_file = open(str('imputation_error_files/nan_imputation_error_file' + label_error_run_name + '.txt'),
                              'a')
        imp_error_file.write('RUNTIME ERROR: Imputation Error as some labels are nan '
                             'ASW metric will be set to -1 to kill off chromosome\n\n\n')
        imp_error_file.write('Affected Chromosome: \n')
        for gene in chromosome:
            imp_error_file.write(str(gene) + ', ')
        imp_error_file.write('\n------------------------------------------------------------------\n\n')
        imp_error_file.close()
        return np.nan
    else:
        return -davies_bouldin_score(X=x_data, labels=labels)



def _correlation_function(x_train_imputed, x_test_imputed, n_comp):
    """Calculate correlation between train and test sets"""
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train_imputed)  # scale data
    x_test_sc = scaler.fit_transform(x_test_imputed)
    cca = CCA(scale=False, n_components=n_comp)  # define CCA
    cca.fit(x_train_sc, x_test_sc)  # fit our scaled data
    x_train_c, x_test_c = cca.transform(x_train_sc, x_test_sc)  # transform our datasests to obtain canonical variates
    return [np.corrcoef(x_train_c[:, i], x_test_c[:, i])[1][0] for i in range(n_comp)]


def _variance_ratio_function(x_train_imputed, x_test_imputed):
    """Calculate Variance ratio function between train and test sets"""
    x_trian_var = np.sum(np.var(x_train_imputed, axis=0))
    x_test_var = np.sum(np.var(x_test_imputed, axis=0))
    return np.min([x_trian_var, x_test_var])/np.max([x_trian_var, x_test_var])


def is_approximately_power_of_two(number, tolerance=1e-9):
    if number <= 0:
        return None
    exponent = 0
    while number > 1:
        number /= 2
        exponent += 1

    # Check if the remaining number is close to 1 within the specified tolerance
    if abs(number - 1) <= tolerance:
        return exponent


def multi_label_cross_validation_parallel(x_data, y_data, model, macro_micro='micro', cv=4.0):
    split_iter = is_approximately_power_of_two(cv)
    if split_iter is None:
        print('ERROR IN CV FOLDS SETTING: For the Multi-Label Clf Run, it is necessary that the number of folds is set'
              'to any value obeying 2^n. Value will be reverted to 4 folds')
        split_iter = 2

    # Split data into cv folds
    x_folds, y_folds = [x_data], [y_data]
    for split_number in range(int(split_iter)):
        if len(x_folds) != len(y_folds):
            print("RUNTIME ERROR: UNEQUAL LENGTHS FOR X_DATA FOLDS AND Y_DATA FOLDS")
        x_replacement_fold_list, y_replacement_fold_list = [], []
        for x_fold, y_fold in zip(x_folds, y_folds):
            x_half1, y_half1, x_half2, y_half2 = iterative_train_test_split(x_fold, y_fold, test_size=0.5)

            x_replacement_fold_list.append(x_half1)
            x_replacement_fold_list.append(x_half2)
            y_replacement_fold_list.append(y_half1)
            y_replacement_fold_list.append(y_half2)

        del x_folds, y_folds
        x_folds, y_folds = x_replacement_fold_list, y_replacement_fold_list

    # Use each fold as a validation set whilst using the rest as a combined training set
    fold_indices = np.array([i for i in range(len(x_folds))])

    def process_fold(val_fold_index):
        mask = np.array([(j in set(np.array([val_fold_index]))) for j in range(len(x_folds))])

        x_valid, y_valid = x_folds[fold_indices[mask][0]], y_folds[fold_indices[mask][0]]

        x_train, y_train = np.array([]), np.array([])
        for n, rem_fold_idx in enumerate(fold_indices[~mask]):
            if n == 0:
                x_train, y_train = x_folds[rem_fold_idx], y_folds[rem_fold_idx]
            else:
                x_train = np.concatenate((x_train, x_folds[rem_fold_idx]), axis=0)
                y_train = np.concatenate((y_train, y_folds[rem_fold_idx]), axis=0)

        model_copy = clone(model)
        model_copy.fit(x_train, y_train)
        y_pred_valid = model_copy.predict(x_valid)

        micro_f1 = f1_score(y_valid, y_pred_valid, average=macro_micro)

        acc_jup = accuracy_score(y_valid[:, 0], y_pred_valid[:, 0])
        acc_rocky = accuracy_score(y_valid[:, 1], y_pred_valid[:, 1])

        f1_jup = f1_score(y_valid[:, 0], y_pred_valid[:, 0])
        f1_rocky = f1_score(y_valid[:, 1], y_pred_valid[:, 1])

        return [micro_f1], [acc_jup, acc_rocky], [f1_jup, f1_rocky]

    scores = Parallel(n_jobs=-1)(delayed(process_fold)(val_fold_index) for val_fold_index in range(len(x_folds)))
    #   Shape of scores: [[micro_f1, acc_lbls (list), f1_lbls (list), ...]
    return scores


class DatasetImputation:

    def __init__(self, x_train, x_test, x_train_missing_i, x_test_missing_i, chromosome, generation_number,
                 features_list, log_absolute_conversion):
        """Impute Missing Values. For the 1st Generation, imputation is done using a mean imputer first. Once imputed,
        clustering is done based on the chromosome's number of clusters and fuzziness parameter. With the centres
        calculated, a jitter term is added to incorporate some noise. The jitter in each feature is scaled with the std
        dev of the population of measured (not including the imputed values) abundances of that feature. Once this is
        done, the code should return the full chromosome and imputed data.

        After Generation 1, the cluster centres are used to calculate a membership degree calculation and the values are
        imputed based on this value. Once this is done, the code should return the full chromosome and imputed data."""
        if generation_number == 1:
            self.full_chromosome, self.x_train_imputed, self.x_test_imputed, self.x_train_cluster_labels, \
             self.x_test_cluster_labels = self._gen1_imputation(
                x_train=x_train, x_test=x_test, partial_chromosome=chromosome, features_list=features_list,
                log_absolute_conversion=log_absolute_conversion)
        else:
            self.full_chromosome, self.x_train_imputed, self.x_test_imputed, self.x_train_cluster_labels, \
             self.x_test_cluster_labels = self._normal_imputation(
                chromosome=chromosome, x_train=x_train, x_test=x_test, x_train_missing_i=x_train_missing_i,
                x_test_missing_i=x_test_missing_i, features_list=features_list,
                log_absolute_conversion=log_absolute_conversion
             )

    def _gen1_imputation(self, x_train, x_test, partial_chromosome, log_absolute_conversion, features_list):
        """ For the 1st Generation, imputation is done using a mean imputer first. Once imputed,
        clustering is done based on the chromosome's number of clusters and fuzziness parameter. With the centres
        calculated, a jitter term is added to incorporate some noise. The jitter in each feature is scaled with the std
        dev of the population of measured (not including the imputed values) abundances of that feature. Once this is
        done, the code should return the full chromosome and imputed data."""

        # Unpack Partial Chromosome
        num_centres = partial_chromosome[0]
        fuzziness = partial_chromosome[1]

        # Mean Imputation
        x_train, x_train_features_std_dev = _mean_imputation(dataset=x_train, features_list=features_list,
                                                             log_absolute_conversion=log_absolute_conversion)
        x_test, x_test_features_std_dev = _mean_imputation(dataset=x_test, features_list=features_list,
                                                           log_absolute_conversion=log_absolute_conversion)

        x_data = np.concatenate([x_train, x_test])
        features_std_dev = np.median([x_train_features_std_dev, x_test_features_std_dev], axis=0)

        # Introduce the Standard Normalisation here where each dataset is fit and transformed
        imputation_scaler = StandardScaler()
        imputation_scaler.fit(x_data)
        scaled_x_data = imputation_scaler.transform(x_data)

        # Fuzzy Clustering And jitter term to centres
        scaled_centres, membership, _, _, _, p, fpc = fuzz.cmeans(data=np.transpose(scaled_x_data), c=int(num_centres),
                                                                  m=fuzziness, error=0.005, maxiter=1000, init=None)
        cluster_labels = np.argmax(membership, axis=0)
        x_train_cluster_labels, x_test_cluster_labels = cluster_labels[:len(x_train)], cluster_labels[len(x_train):]

        # Inverse transform the dataset and centres back to original absolute values
        centres = imputation_scaler.inverse_transform(scaled_centres)

        if p >= 999:
            sys.exit("RUNTIME ERROR: FUZZY CLUSTERING FAILED TO CONVERGE FOR CHROMOSOME {}".format(partial_chromosome))

        full_chromosome = []
        for centre in centres:
            centre += features_std_dev*np.random.uniform(low=-1., high=1., size=(len(features_std_dev)))
            full_chromosome = np.concatenate([full_chromosome, centre])
        full_chromosome = np.concatenate([full_chromosome, partial_chromosome[1:]])

        return full_chromosome, x_train, x_test, x_train_cluster_labels, x_test_cluster_labels

    def _normal_imputation(self, chromosome, x_train, x_test, x_train_missing_i, x_test_missing_i,
                           log_absolute_conversion, features_list):
        num_features = x_train.shape[1]
        centres = chromosome[:-9]
        centres = np.array(centres).reshape(int(len(centres)/num_features), int(num_features))
        fuzziness = chromosome[-9]

        x_train_cluster_labels, x_test_cluster_labels = [], []

        if log_absolute_conversion:
            x_train = _log_to_absolute_feature_conversion(x_train, features_list)
            x_test = _log_to_absolute_feature_conversion(x_test, features_list)
            centres = _log_to_absolute_feature_conversion(centres, features_list)

        # Introduce the Standard Normalisation here where each dataset is fit and transformed
        normal_imp_scaler = StandardScaler()
        normal_imp_scaler.fit(x_train)
        x_train_scaled = normal_imp_scaler.transform(x_train)
        x_test_scaled = normal_imp_scaler.transform(x_test)
        centres_scaled = normal_imp_scaler.transform(centres)

        # Find feature index of f_disk in features_list if it is included
        f_disk_idx = None
        if 'f_disk' in features_list:
            f_disk_idx = features_list.index('f_disk')

        for j, datapoint in enumerate(x_train_scaled):
            memberships = membership_calculation(datapoint=datapoint, centres=centres_scaled, fuzziness=fuzziness)
            x_train_cluster_labels.append(np.argmax(memberships))
            for i in x_train_missing_i[j]:
                if 'f_disk' in features_list:
                    if i == f_disk_idx:
                        datapoint[i] = np.round(membership_imputation(feature_index=i, memberships=memberships,
                                                                      centres=centres_scaled))
                    else:
                        datapoint[i] = membership_imputation(feature_index=i, memberships=memberships,
                                                             centres=centres_scaled)
                else:
                    datapoint[i] = membership_imputation(feature_index=i, memberships=memberships,
                                                         centres=centres_scaled)

        for j, datapoint in enumerate(x_test_scaled):
            memberships = membership_calculation(datapoint=datapoint, centres=centres_scaled, fuzziness=fuzziness)
            x_test_cluster_labels.append(np.argmax(memberships))
            for i in x_test_missing_i[j]:
                if 'f_disk' in features_list:
                    if i == f_disk_idx:
                        datapoint[i] = np.round(membership_imputation(feature_index=i, memberships=memberships,
                                                                      centres=centres_scaled))
                    else:
                        datapoint[i] = membership_imputation(feature_index=i, memberships=memberships,
                                                             centres=centres_scaled)
                else:
                    datapoint[i] = membership_imputation(feature_index=i, memberships=memberships,
                                                         centres=centres_scaled)

        # Inverse transformed imputed datasets back to original absolute values
        x_train_imputed = normal_imp_scaler.inverse_transform(x_train_scaled)
        x_test_imputed = normal_imp_scaler.inverse_transform(x_test_scaled)

        if log_absolute_conversion:
            x_train_imputed = _absolute_to_log_feature_conversion(x_train_imputed, features_list)
            x_test_imputed = _absolute_to_log_feature_conversion(x_test_imputed, features_list)

        return chromosome, x_train_imputed, x_test_imputed, x_train_cluster_labels, x_test_cluster_labels


class FitnessCalculation:
    def __init__(self, chromosome, x_train_imputed, y_train, x_test_imputed, x_train_cluster_labels,
                 x_test_cluster_labels, cluster_val_fn_choice, cross_val, clf_metric, num_datasubsamples,
                 label_error_run_name):
        """Calculates fitness values for each chromosome for both objective functions.

        Objective function 1: Imputation
        Objective function 2,3: Classification
        """

        self.cluster_centres = chromosome[:-9]
        self.fuzziness = chromosome[-9]
        self.xgb_parameters = chromosome[-8:]

        self.fitness_1 = self._objective_fn_1(x_train_imputed=x_train_imputed, x_test_imputed=x_test_imputed,
                                              cluster_val_fn_choice=cluster_val_fn_choice,
                                              x_train_cluster_labels=x_train_cluster_labels,
                                              x_test_cluster_labels=x_test_cluster_labels, chromosome=chromosome,
                                              label_error_run_name=label_error_run_name)

        self.fitness_2, self.fitness_3, self.accuracy_jup, self.accuracy_rocky, \
        self.f1_jup, self.f1_rocky  \
            = self._objective_fn_2(x_train_imputed=x_train_imputed, y_train=y_train,
                                   xgb_hyperparams=self.xgb_parameters,
                                   cross_val=cross_val, clf_metric=clf_metric,
                                   num_datasubsamples=num_datasubsamples)

    def _objective_fn_1(self, x_train_imputed, x_test_imputed, x_train_cluster_labels, x_test_cluster_labels,
                        cluster_val_fn_choice, chromosome, label_error_run_name):
        """Calculate the fitness dependent on which cluster validity function is chosen
           cluster_val_fn_choice=1 : Average Silhouette Width
           cluster_val_fn_choice=2 : Correlation Function
           cluster_val_fn_choice=3 : Variance Ratio Function
           cluster_val_fn_choice=4 : Davis-Bouldin Score
           """
        if cluster_val_fn_choice == 1:
            return _average_silhouette_width(x_train_imputed=x_train_imputed, x_test_imputed=x_test_imputed,
                                             x_train_cluster_labels=x_train_cluster_labels,
                                             x_test_cluster_labels=x_test_cluster_labels, chromosome=chromosome,
                                             label_error_run_name=label_error_run_name)

        elif cluster_val_fn_choice == 2:
            corrfn = _correlation_function(x_train_imputed=x_train_imputed, x_test_imputed=x_test_imputed, n_comp=1)
            return corrfn[0]
        elif cluster_val_fn_choice == 3:
            return _variance_ratio_function(x_train_imputed, x_test_imputed)
        elif cluster_val_fn_choice == 4:
            return _davies_bouldin_score_calculation(x_train_imputed=x_train_imputed, x_test_imputed=x_test_imputed,
                                                     x_train_cluster_labels=x_train_cluster_labels,
                                                     x_test_cluster_labels=x_test_cluster_labels, chromosome=chromosome,
                                                     label_error_run_name=label_error_run_name)
        else:
            sys.exit("RUNTIME ERROR: Choice of cluster validity function is incorrect")

    def _objective_fn_2(self, x_train_imputed, y_train, xgb_hyperparams, cross_val, clf_metric, num_datasubsamples):
        n_est, max_depth, lr, subsample, colsample, gamma, reg_alpha, reg_lambda = xgb_hyperparams

        data_subsamples = _dataset_subsampling(x_data=x_train_imputed, y_data=y_train,
                                               num_subsamples=num_datasubsamples, balancesplit=0.6)

        clf = Pipeline([
            ("std_scaler", StandardScaler()),
            ("xgb_clf", MultiOutputClassifier(xgb.XGBClassifier(objective='binary:logistic', n_estimators=int(n_est),
                                                                max_depth=int(max_depth) , learning_rate=float(lr),
                                                                subsample=float(subsample),
                                                                colsample_bytree=float(colsample), gamma=float(gamma),
                                                                reg_alpha=float(reg_alpha),
                                                                reg_lambda=float(reg_lambda)
                                                                )))
        ])

        scores_clf_metric = np.array([])
        scores_accuracy_jup, scores_accuracy_rocky = np.array([]), np.array([])
        scores_f1_jup, scores_f1_rocky = np.array([]), np.array([])

        for sub_sample in data_subsamples:
            scores_subsample = multi_label_cross_validation_parallel(sub_sample[0], sub_sample[1], clf,
                                                                     macro_micro='micro', cv=cross_val)

            for fold_scores in scores_subsample:
                scores_clf_metric = np.concatenate((scores_clf_metric, fold_scores[0]))
                scores_accuracy_jup = np.append(scores_accuracy_jup, fold_scores[1][0])
                scores_accuracy_rocky = np.append(scores_accuracy_rocky, fold_scores[1][1])
                scores_f1_jup = np.append(scores_f1_jup, fold_scores[2][0])
                scores_f1_rocky = np.append(scores_f1_rocky, fold_scores[2][1])

        if clf_metric == 'accuracy':
            return \
                np.mean(scores_accuracy_jup), np.mean(scores_accuracy_rocky), \
                [np.mean(scores_accuracy_jup), np.std(scores_accuracy_jup)], \
                [np.mean(scores_accuracy_rocky), np.std(scores_accuracy_rocky)], \
                [np.mean(scores_f1_jup), np.std(scores_f1_jup)], \
                [np.mean(scores_f1_rocky), np.std(scores_f1_rocky)]
        elif clf_metric == 'f1':
            return \
                np.mean(scores_f1_jup), np.mean(scores_f1_rocky), \
                [np.mean(scores_accuracy_jup), np.std(scores_accuracy_jup)], \
                [np.mean(scores_accuracy_rocky), np.std(scores_accuracy_rocky)], \
                [np.mean(scores_f1_jup), np.std(scores_f1_jup)], \
                [np.mean(scores_f1_rocky), np.std(scores_f1_rocky)]
        else:
            print("ERROR IN CLF_METRIC SETTING: FITNESS METRIC SET TO ACCURACY AS DEFAULT")
            return \
                np.mean(scores_accuracy_jup), np.mean(scores_accuracy_rocky), \
                [np.mean(scores_accuracy_jup), np.std(scores_accuracy_jup)], \
                [np.mean(scores_accuracy_rocky), np.std(scores_accuracy_rocky)], \
                [np.mean(scores_f1_jup), np.std(scores_f1_jup)], \
                [np.mean(scores_f1_rocky), np.std(scores_f1_rocky)]
