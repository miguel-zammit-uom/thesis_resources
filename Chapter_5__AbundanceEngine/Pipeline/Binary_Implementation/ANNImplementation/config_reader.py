import numpy as np
from configparser import ConfigParser as cP


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class ConfigReader:

    def __init__(self, configfile):
        """Load Configuration settings from config file"""
        self.config = cP()
        self.config.read(configfile)

        # Genetic Algorithm Settings
        self.max_generations = np.int(self.config.get('Genetic Algorithm Settings', 'max_generations'))
        self.pop_size = np.int(self.config.get('Genetic Algorithm Settings', 'pop_size'))
        self.imputation_threshold = np.float(self.config.get('Genetic Algorithm Settings',
                                                             'imputation_obj_val_threshold'))
        self.clf_threshold = np.float(self.config.get('Genetic Algorithm Settings', 'clf_obj_val_threshold'))
        self.rndm_seed = np.float(self.config.get('Genetic Algorithm Settings', 'rndm_seed'))
        self.num_replacement_fronts = np.int(self.config.get('Genetic Algorithm Settings', 'num_replacement_fronts'))
        self.c_r = np.int(self.config.get('Genetic Algorithm Settings', 'c_r'))
        self.run_name = np.str(self.config.get('Genetic Algorithm Settings', 'run_name'))
        self.generations_stats_out_txt = np.str(self.config.get('Genetic Algorithm Settings',
                                                                'generations_stats_out_txt')) + '_' + self.run_name + '.txt'

        # Dataset Settings
        self.dataset_path = np.str(self.config.get('Dataset Settings', 'dataset_path'))
        self.dataset_name = np.str(self.config.get('Dataset Settings', 'dataset_name')).replace(' ', '_')
        self.element_features_list = np.str(self.config.get('Dataset Settings', 'element_features_list')).split(",")
        self.log_absolute_conversion = str2bool(np.str(self.config.get('Dataset Settings', 'log_absolute_conversion')))

        # Imputation Settings
        self.max_clusters = np.int(self.config.get('Imputation Settings', 'max_clusters'))
        self.fuzziness_min = np.float(self.config.get('Imputation Settings', 'fuzziness_min'))
        self.fuzziness_max = np.float(self.config.get('Imputation Settings', 'fuzziness_max'))
        self.cluster_validity_fn = np.int(self.config.get('Imputation Settings', 'cluster_validity_fn'))

        # Classification Settings
        self.cross_val = np.int(self.config.get('Classification Settings', 'cross_validation'))
        self.clf_metric = np.str(self.config.get('Classification Settings', 'clf_metric'))
        self.num_datasubsamples = np.int(self.config.get('Classification Settings', 'num_datasubsamples'))
        self.num_hdn_layers_min = np.int(self.config.get('Classification Settings', 'num_hdn_layers_min'))
        self.num_hdn_layers_max = np.int(self.config.get('Classification Settings', 'num_hdn_layers_max'))
        self.num_neurons_first_min = np.int(self.config.get('Classification Settings', 'num_neurons_first_min'))
        self.num_neurons_first_max = np.int(self.config.get('Classification Settings', 'num_neurons_first_max'))
        self.reg_l1_smallest_order = np.int(self.config.get('Classification Settings', 'reg_l1_smallest_order'))
        self.reg_l2_smallest_order = np.int(self.config.get('Classification Settings', 'reg_l2_smallest_order'))
        self.dropout_min = np.float(self.config.get('Classification Settings', 'dropout_min'))
        self.dropout_max = np.float(self.config.get('Classification Settings', 'dropout_max'))
        self.num_epochs_min = np.int(self.config.get('Classification Settings', 'num_epochs_min'))
        self.num_epochs_max = np.int(self.config.get('Classification Settings', 'num_epochs_max'))

        # Accepted Ranges Dictionary
        self.accepted_ranges = {
            "fuzz_min": self.fuzziness_min,
            "fuzz_max": self.fuzziness_max,
            "num_hdn_layers_min": self.num_hdn_layers_min,
            "num_hdn_layers_max": self.num_hdn_layers_max,
            "num_neurons_first_min": self.num_neurons_first_min,
            "num_neurons_first_max": self.num_neurons_first_max,
            "reg_l1_smallest_order": self.reg_l1_smallest_order,
            "reg_l2_smallest_order": self.reg_l2_smallest_order,
            "dropout_min": self.dropout_min,
            "dropout_max": self.dropout_max,
            "num_epochs_min": self.num_epochs_min,
            "num_epochs_max": self.num_epochs_max
        }
