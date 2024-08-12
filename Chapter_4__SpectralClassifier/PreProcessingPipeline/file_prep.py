import os
import warnings
import pandas as pd
import logging
from tqdm import tqdm


class FileMetadataPrep:

    def __init__(self, spectra_root, training_label_file, spectra_list_csv, training_label_file_index='Star'):
        """Load spectra filenames and prepare dictionary of relevant information"""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='path/to/logging/directory'
                                     'FileMetadataPrep_runtime_messages.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')
        self.spectra_file_list = os.listdir(spectra_root)

        self.training_label_file = pd.read_csv(training_label_file, index_col=training_label_file_index)
        self.spectra_list_csv = pd.read_csv(spectra_list_csv)

        # Cleanup of filenames in spectra_list_csv
        new_obs_id_col = []
        for row in self.spectra_list_csv.iterrows():
            fixed_id = row[1]['obs_id'].replace('-', '_')
            fixed_id = fixed_id.replace(':', '_')
            fixed_id = fixed_id.replace('ADP.', 'ADP_')
            new_obs_id_col.append(fixed_id + '.fits')
        self.spectra_list_csv['new_obs_id'] = new_obs_id_col

        self.spectra_master_list = list()
        # ['Filename', 'InvestigatorID', 'Target Name', 'Jupiter Label', 'Number of Planets']

        # Switched to True whenever a target does not have a listed label and needs to be dumped from dataset
        self.label_error = False

        for spectrum in tqdm(self.spectra_file_list, desc='Loading Spectra Metadata'):
            list_entry = []

            # Get name of the target name from filename
            self.target_name = self.spectra_list_csv.loc[
                self.spectra_list_csv['new_obs_id'] == spectrum
                ]['fov'].values[0]
            self.investigator = self.spectra_list_csv.loc[
                self.spectra_list_csv['new_obs_id'] == spectrum
                ]['source'].values[0]


            self.label, self.num_planets, self.label_error = self._training_label_check(self.target_name)
            if not self.label_error:
                # Place information in dictionary
                list_entry.extend([spectrum, self.investigator, self.target_name, self.label, self.num_planets])
            else:
                pass
            if len(list_entry) == 5:    # If entry is complete and label is found
                self.spectra_master_list.append(list_entry)

    def _training_label_check(self, target_name):
        """Checks whether target star of spectrum is listed in training label csv and discards those that are not"""
        if target_name in self.training_label_file.index:
            label, num_planets, = self.training_label_file.loc[str(target_name)]['Jupiter Host'], \
                                  self.training_label_file.loc[str(target_name)]['Number of Planets']
            return label, num_planets, False
        else:
            logging.warning('Label for target {} has not been constrained. Spectrum will be omitted from the '
                            'dataset'.format(str(target_name)))
            return None, None, True
