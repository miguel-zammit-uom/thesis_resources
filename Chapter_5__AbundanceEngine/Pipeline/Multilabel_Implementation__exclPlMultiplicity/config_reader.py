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
        self.max_generations = int(self.config.get('Genetic Algorithm Settings', 'max_generations'))
        self.pop_size = int(self.config.get('Genetic Algorithm Settings', 'pop_size'))
        self.imputation_threshold = float(self.config.get('Genetic Algorithm Settings',
                                                             'imputation_obj_val_threshold'))
        self.clf_threshold = float(self.config.get('Genetic Algorithm Settings', 'clf_obj_val_threshold'))
        self.rndm_seed = float(self.config.get('Genetic Algorithm Settings', 'rndm_seed'))
        self.num_replacement_fronts = int(self.config.get('Genetic Algorithm Settings', 'num_replacement_fronts'))
        self.c_r = int(self.config.get('Genetic Algorithm Settings', 'c_r'))
        self.run_name = str(self.config.get('Genetic Algorithm Settings', 'run_name'))
        self.generations_stats_out_txt = str(self.config.get('Genetic Algorithm Settings',
                                                             'generations_stats_out_txt')) + '_' + self.run_name + '.txt'

        # Dataset Settings
        self.dataset_path = str(self.config.get('Dataset Settings', 'dataset_path'))
        self.dataset_name = str(self.config.get('Dataset Settings', 'dataset_name')).replace(' ', '_')
        self.element_features_list = str(self.config.get('Dataset Settings', 'element_features_list')).split(",")
        self.log_absolute_conversion = str2bool(str(self.config.get('Dataset Settings', 'log_absolute_conversion')))

        # Imputation Settings
        self.max_clusters = int(self.config.get('Imputation Settings', 'max_clusters'))
        self.fuzziness_min = float(self.config.get('Imputation Settings', 'fuzziness_min'))
        self.fuzziness_max = float(self.config.get('Imputation Settings', 'fuzziness_max'))
        self.cluster_validity_fn = int(self.config.get('Imputation Settings', 'cluster_validity_fn'))

        # Classification Settings
        self.cross_val = int(self.config.get('Classification Settings', 'cross_validation'))
        self.clf_metric = str(self.config.get('Classification Settings', 'clf_metric'))
        self.num_datasubsamples = int(self.config.get('Classification Settings', 'num_datasubsamples'))
        self.n_est_min = int(self.config.get('Classification Settings', 'n_est_min'))
        self.n_est_max = int(self.config.get('Classification Settings', 'n_est_max'))
        self.max_depth_min = int(self.config.get('Classification Settings', 'max_depth_min'))
        self.max_depth_max = int(self.config.get('Classification Settings', 'max_depth_max'))
        self.lr_smallest_order = int(self.config.get('Classification Settings', 'lr_smallest_order'))
        self.subsample_min = float(self.config.get('Classification Settings', 'subsample_min'))
        self.subsample_max = float(self.config.get('Classification Settings', 'subsample_max'))
        self.col_sample_min = float(self.config.get('Classification Settings', 'col_sample_min'))
        self.col_sample_max = float(self.config.get('Classification Settings', 'col_sample_max'))
        self.gamma_min = float(self.config.get('Classification Settings', 'gamma_min'))
        self.gamma_max = float(self.config.get('Classification Settings', 'gamma_max'))
        self.reg_alpha_smallest_order = int(self.config.get('Classification Settings', 'reg_alpha_smallest_order'))
        self.reg_lambda_smallest_order = int(self.config.get('Classification Settings', 'reg_lambda_smallest_order'))

        # Accepted Ranges Dictionary
        self.accepted_ranges = {
            "fuzz_min": self.fuzziness_min,
            "fuzz_max": self.fuzziness_max,
            "n_est_min": self.n_est_min,
            "n_est_max": self.n_est_max,
            "max_depth_min": self.max_depth_min,
            "max_depth_max": self.max_depth_max,
            "lr_smallest_order": self.lr_smallest_order,
            "subsample_min": self.subsample_min,
            "subsample_max": self.subsample_max,
            "col_sample_min": self.col_sample_min,
            "col_sample_max": self.col_sample_max,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "reg_alpha_smallest_order": self.reg_alpha_smallest_order,
            "reg_lambda_smallest_order": self.reg_lambda_smallest_order
        }
