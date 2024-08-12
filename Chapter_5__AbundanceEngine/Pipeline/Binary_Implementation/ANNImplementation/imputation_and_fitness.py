import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, recall_score, f1_score, \
    precision_score
import sys
import tensorflow as tf
from tensorflow import keras
import skfuzzy.cluster as fuzz
from joblib import Parallel, delayed
import multiprocessing
import logging
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



def _build_ann_model(num_features, num_hdn_layers, num_neurons_first, reg_l1, reg_l2, dropout, act_fn_choice,
                     optimizer_choice, clf_metric):

    activation = None
    if act_fn_choice == 1:
        activation = 'selu'
    elif act_fn_choice == 2:
        activation = 'elu'
    elif act_fn_choice == 3:
        activation = 'relu'
    elif act_fn_choice == 4:
        activation = 'tanh'

    optimizer = None
    if optimizer_choice == 1:
        optimizer = 'adam'
    elif optimizer_choice == 2:
        optimizer = 'adagrad'
    elif optimizer_choice == 3:
        optimizer = 'nadam'
    elif optimizer_choice == 4:
        optimizer = 'sgd'

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(num_features,)),
        keras.layers.Dropout(dropout/2),    # dropout rate for input layer is halved so that it avoids creating a
                                            # bottleneck in the beginning of the network
        keras.layers.BatchNormalization()])

    for hl_count in range(int(num_hdn_layers)):
        n_neurons = num_neurons_first*(2**hl_count)

        if activation == 'selu':
            model.add(keras.layers.Dense(n_neurons, kernel_initializer='lecun_normal', use_bias=True,
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)))
        elif activation == 'elu' or activation == 'relu':
            model.add(keras.layers.Dense(n_neurons, kernel_initializer='he_normal', use_bias=True,
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)))
        elif activation == 'tanh':
            model.add(keras.layers.Dense(n_neurons, kernel_initializer='glorot_normal', use_bias=True,
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)))
        else:
            sys.exit('ERROR IN CHOICE OF ACTIVATION FUNCTION')

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(str(activation)))

        # Incorporating a dropout layer after every 5 hidden layers, just to incorporate more regularisation
        if hl_count % 5 == 0 and (0 < hl_count < num_hdn_layers-1):
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=str(optimizer))
    return model

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


def _dataset_subsampling(x_data, y_data, num_subsamples, balancesplit=0.6):
    """Subsamples the 0 label subset to decrease the imbalance of the dataset. balancesplit sets the fraction of
    the dataset with a 0 label. """

    # Drop the final feature as this is the Jupiter Host Label
    x_data = np.delete(x_data, -1, axis=1)

    num_transformer = Pipeline([
        ("std_scaler", StandardScaler()),
    ])
    x_data = num_transformer.fit_transform(x_data)

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

        for j, datapoint in enumerate(x_train_scaled):
            memberships = membership_calculation(datapoint=datapoint, centres=centres_scaled, fuzziness=fuzziness)
            x_train_cluster_labels.append(np.argmax(memberships))
            for i in x_train_missing_i[j]:
                datapoint[i] = membership_imputation(feature_index=i, memberships=memberships, centres=centres_scaled)

        for j, datapoint in enumerate(x_test_scaled):
            memberships = membership_calculation(datapoint=datapoint, centres=centres_scaled, fuzziness=fuzziness)
            x_test_cluster_labels.append(np.argmax(memberships))
            for i in x_test_missing_i[j]:
                datapoint[i] = membership_imputation(feature_index=i, memberships=memberships, centres=centres_scaled)

        # Inverse transformed imputed datasets back to original absolute values
        x_train_imputed = normal_imp_scaler.inverse_transform(x_train_scaled)
        x_test_imputed = normal_imp_scaler.inverse_transform(x_test_scaled)

        if log_absolute_conversion:
            x_train_imputed = _absolute_to_log_feature_conversion(x_train_imputed, features_list)
            x_test_imputed = _absolute_to_log_feature_conversion(x_test_imputed, features_list)

        return chromosome, x_train_imputed, x_test_imputed, x_train_cluster_labels, x_test_cluster_labels


def _cross_validation_custom_fn(model, num_epochs, X, y, train, test):

    model.fit(X[train], y[train], epochs=int(num_epochs), verbose=0,
              batch_size=128)

    pred = np.round(model.predict(X[test], verbose=0))
    accuracy = accuracy_score(y[test], pred)
    recall = recall_score(y[test], pred)
    f1 = f1_score(y[test], pred)
    precision = precision_score(y[test], pred)

    return accuracy, f1, recall, precision


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
        self.ann_parameters = chromosome[-8:]

        self.fitness_1 = self._objective_fn_1(x_train_imputed=x_train_imputed, x_test_imputed=x_test_imputed,
                                              cluster_val_fn_choice=cluster_val_fn_choice,
                                              x_train_cluster_labels=x_train_cluster_labels,
                                              x_test_cluster_labels=x_test_cluster_labels, chromosome=chromosome,
                                              label_error_run_name=label_error_run_name)
        self.fitness_2, self.accuracy_score, self.f1_score, self.recall_score, self.precision_score \
            = self._objective_fn_2(x_train_imputed=x_train_imputed, y_train=y_train,
                                   ann_hyperparams=self.ann_parameters,
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

    def _objective_fn_2(self, x_train_imputed, y_train, ann_hyperparams, cross_val, clf_metric, num_datasubsamples):
        global model
        num_hdn_layers, num_neurons_first, reg_l1, reg_l2, dropout, act_fn_choice, optimizer_choice, num_epochs  \
            = ann_hyperparams

        # Set TensorFlow logging level to ERROR
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        data_subsamples = _dataset_subsampling(x_data=x_train_imputed, y_data=y_train,
                                               num_subsamples=num_datasubsamples, balancesplit=0.6)

        scores_clf_metric, scores_accuracy, scores_f1, scores_recall, scores_precision \
            = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        model = _build_ann_model(num_features=x_train_imputed.shape[1]-1, num_hdn_layers=num_hdn_layers,
                                 num_neurons_first=num_neurons_first, reg_l1=reg_l1, reg_l2=reg_l2,
                                 dropout=dropout, act_fn_choice=act_fn_choice,
                                 optimizer_choice=optimizer_choice, clf_metric=clf_metric)
        for sub_sample in data_subsamples:

            X = sub_sample[0].reshape((sub_sample[0].shape[0], sub_sample[0].shape[1]))
            y = np.array(sub_sample[1])

            kfold = StratifiedKFold(n_splits=cross_val, shuffle=True)

            model_copies = [copy.copy(model) for i in range(cross_val)]

            scores = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(
                delayed(_cross_validation_custom_fn)(model_copies[i], num_epochs, X, y, train_index, test_index)
                for i, (train_index, test_index) in enumerate(kfold.split(X, y))
            )

            accuracy, f1, recall, precision = np.transpose(np.array(scores))

            if np.max(f1) >= 1 or np.max(accuracy) >= 1 or np.max(recall) > 1 or np.max(precision) > 1:
                print("{} Chromosome: ERROR IN METRIC CALCULATION Acc: {}, f1: {}, Pre:{}, Recall:{}\n Fitness to 0 "
                      "to remove tainted calculations".format(
                    ann_hyperparams, np.max(accuracy), np.max(f1), np.max(recall), np.max(precision)))
                accuracy, f1, recall, precision = np.zeros(shape=accuracy.shape), np.zeros(shape=f1.shape), \
                                                  np.zeros(shape=recall.shape), np.zeros(shape=precision.shape)

            scores_accuracy = np.concatenate([scores_accuracy, accuracy])
            scores_f1 = np.concatenate([scores_f1, f1])
            scores_recall = np.concatenate([scores_recall, recall])
            scores_precision = np.concatenate([scores_precision, precision])

            del model_copies
            keras.backend.clear_session()

        del model

        if clf_metric == 'accuracy':
            scores_clf_metric = np.copy(scores_accuracy)
        elif clf_metric == 'f1':
            scores_clf_metric = np.copy(scores_f1)
        elif clf_metric == 'recall':
            scores_clf_metric = np.copy(scores_recall)
        elif clf_metric == 'precision':
            scores_clf_metric = np.copy(scores_precision)

        return \
            np.mean(scores_clf_metric), \
            [np.mean(scores_accuracy), np.std(scores_accuracy)], \
            [np.mean(scores_f1), np.std(scores_f1)], \
            [np.mean(scores_recall), np.std(scores_recall)], \
            [np.mean(scores_precision), np.std(scores_precision)]

