import numpy as np
from configparser import ConfigParser as cP
import os


class ConfigReader:

    def __init__(self, configfile):
        """Load settings from given configuration file"""
        self.config = cP()
        self.config.read(configfile)

        self.parent_directory = np.str(self.config.get('File Configuration', 'parent_directory'))
        self.spectra_root = np.str(os.path.join(self.parent_directory, self.config.get('File Configuration',
                                                                                       'spectra_root')))
        self.training_labels = np.bool(self.config.get('File Configuration',
                                                       'training_labels'))
        if self.training_labels:
            self.training_label_file = np.str(os.path.join(self.parent_directory,
                                                           self.config.get('File Configuration',
                                                                           'training_label_file')
                                                           ))
        else:
            self.training_label_file = None
        self.spectra_list_csv = np.str(os.path.join(self.parent_directory,
                                                    self.config.get('File Configuration',
                                                                    'spectra_list_csv')))

        self.data_destination_path = np.str(self.config.get('Output File Configuration',
                                                            'data_destination_path'))
        self.custom_directory_switch = np.bool(self.config.get('Output File Configuration',
                                                               'custom_directory_switch'))
        self.custom_directory = np.str(self.config.get('Output File Configuration',
                                                       'custom_directory'))

        self.final_resolution = np.int(self.config.get('Output Parameters',
                                                       'final_resolution'))
        self.max_points_to_trim = np.int(self.config.get('Output Parameters',
                                                         'max_points_to_trim'))
        self.lower_limit_FEROS = np.int(self.config.get('Output Parameters',
                                                        'lower_limit_FEROS'))
        self.upper_limit_FEROS = np.int(self.config.get('Output Parameters',
                                                        'upper_limit_FEROS'))
        self.num_of_cnn_bins = np.int(self.config.get('Output Parameters',
                                                      'num_of_cnn_bins'))
        self.equipartition_bins = np.bool(self.config.get('Output Parameters',
                                                          'equipartition_bins'))
