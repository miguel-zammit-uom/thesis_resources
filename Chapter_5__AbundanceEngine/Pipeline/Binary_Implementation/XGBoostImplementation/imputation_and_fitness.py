import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys
import skfuzzy.cluster as fuzz
import xgboost as xgb


def _log_to_absolute_feature_conversion(dataset, features_list):
    non_log_features = 'f_ra,f_dec,f_x,f_y,f_z,f_dist,f_disk,f_spec,f_vmag,f_bv,f_u,f_v,f_w,f_teff,f_mass,f_radius,' \
                       'f_name,f_p,f_m_p,f_e,f_a,f_r_p,f_min_m_p,Jupiter_Host'.split(',')
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
                       'f_name,f_p,f_m_p,f_e,f_a,f_r_p,f_min_m_p,Jupiter_Host'.split(',')
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


def _dataset_subsampling(x_data, y_data, num_subsamples, balancesplit=0.816):
    """Subsamples the 0 label subset to decrease the imbalance of the dataset. balancesplit sets the fraction of
    the dataset with a 0 label. """

    # Drop the final feature as this is the Jupiter Host Label
    x_data = np.delete(x_data, -1, axis=1)
    x_data_hosts, y_data_hosts = x_data[np.argwhere(y_data == 1)], y_data[np.argwhere(y_data == 1)]
    x_data_comparisons, y_data_comparisons = x_data[np.argwhere(y_data == 0)], y_data[np.argwhere(y_data == 0)]


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
                                y_data_sample.reshape((y_data_sample.shape[0],))])
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
                           's,f_name,f_p,f_m_p,f_e,f_a,f_r_p,f_min_m_p,Jupiter_Host'.split(',')
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


class DatasetImputation:

    def __init__(self, x_train, x_test, x_train_missing_i, x_test_missing_i, chromosome, generation_number,
                 features_list, log_absolute_conversion):
        """Impute Missing Values. For the 1st Generation, imputation is done using a mean imputer first. Once imputed,
        clustering is done based on the chromosome's number of clusters and fuzziness parameter. With the centres
        calculated, a jitter term is added to incorporate some noise. The jitter in each feature is scaled with the std
        dev of the population of measured (not including the imputed values) abundances of that feature. Once this is
        done, the code should return the full chromosome and imputed data.

        After Generation 1, the cluster centres are used to calculate a membership degree calculation and the values are
        imputed based on this value."""
        
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
                    datapoint[i] = membership_imputation(feature_index=i, memberships=memberships, centres=centres_scaled)

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
        Objective function 2: Classification
        
        """

        self.cluster_centres = chromosome[:-9]
        self.fuzziness = chromosome[-9]
        self.xgb_parameters = chromosome[-8:]

        self.fitness_1 = self._objective_fn_1(x_train_imputed=x_train_imputed, x_test_imputed=x_test_imputed,
                                              cluster_val_fn_choice=cluster_val_fn_choice,
                                              x_train_cluster_labels=x_train_cluster_labels,
                                              x_test_cluster_labels=x_test_cluster_labels, chromosome=chromosome,
                                              label_error_run_name=label_error_run_name)
        self.fitness_2, self.accuracy_score, self.f1_score, self.recall_score, self.precision_score \
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
                                               num_subsamples=num_datasubsamples, balancesplit=0.816)

        clf = Pipeline([
            ("std_scaler", StandardScaler()),
            ("xgb_clf", xgb.XGBClassifier(objective='binary:logistic', n_estimators=int(n_est), max_depth=int(max_depth)
                                          , learning_rate=float(lr), subsample=float(subsample),
                                          colsample_bytree=float(colsample), gamma=float(gamma),
                                          reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda)))
        ])

        scores_clf_metric, scores_accuracy, scores_f1, scores_recall, scores_precision \
            = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        for sub_sample in data_subsamples:
            if clf_metric not in ['accuracy', 'recall', 'precision', 'f1']:
                scores_subsample = cross_validate(clf, sub_sample[0], sub_sample[1], cv=cross_val,
                                                  scoring=[str(clf_metric), 'accuracy', 'recall', 'precision', 'f1'],
                                                  n_jobs=8)
            else:
                scores_subsample = cross_validate(clf, sub_sample[0], sub_sample[1], cv=cross_val,
                                                  scoring=['accuracy', 'recall', 'precision', 'f1'], n_jobs=8)


            scores_clf_metric = np.concatenate((scores_clf_metric, scores_subsample[str('test_'+clf_metric)]))
            scores_accuracy = np.concatenate((scores_accuracy, scores_subsample['test_accuracy']))
            scores_f1 = np.concatenate((scores_accuracy, scores_subsample['test_f1']))
            scores_recall = np.concatenate((scores_accuracy, scores_subsample['test_recall']))
            scores_precision = np.concatenate((scores_accuracy, scores_subsample['test_recall']))

        return \
            np.mean(scores_clf_metric), \
            [np.mean(scores_accuracy), np.std(scores_accuracy)], \
            [np.mean(scores_f1), np.std(scores_f1)], \
            [np.mean(scores_recall), np.std(scores_recall)], \
            [np.mean(scores_precision), np.std(scores_precision)]
