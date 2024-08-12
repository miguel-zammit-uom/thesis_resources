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
        self.flexibility_min = np.float(self.config.get('Classification Settings', 'flexibility_min'))
        self.flexibility_max = np.float(self.config.get('Classification Settings', 'flexibility_max'))
        self.gamma_min = np.float(self.config.get('Classification Settings', 'gamma_min'))
        self.gamma_max = np.float(self.config.get('Classification Settings', 'gamma_max'))
        self.r_min = np.float(self.config.get('Classification Settings', 'r_min'))
        self.r_max = np.float(self.config.get('Classification Settings', 'r_max'))
        self.d_min = np.int(self.config.get('Classification Settings', 'd_min'))
        self.d_max = np.int(self.config.get('Classification Settings', 'd_max'))

        # Accepted Ranges Dictionary
        self.accepted_ranges = {
            "fuzz_min": self.fuzziness_min,
            "fuzz_max": self.fuzziness_max,
            "flex_min": self.flexibility_min,
            "flex_max": self.flexibility_max,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "d_min": self.d_min,
            "d_max": self.d_max
        }
