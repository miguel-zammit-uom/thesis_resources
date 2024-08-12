import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _missing_entries_i(dataset):
    missing_entries_i = []
    for entry_i in range(dataset.shape[0]):
        missing_values_index = np.argwhere(np.isnan(dataset[entry_i, :])).flatten()
        missing_entries_i.append(missing_values_index.tolist())
    return missing_entries_i


class DataPreparation:

    def __init__(self, dataset_path, element_features_list):
        """Load the dataset and prepare it for use in GA"""

        self.traindata = pd.read_csv(dataset_path)

        # features list will include the host label. This will be included for clustering and imputation, but removed
        # just before the classification stage. This should give the imputation process further information regarding
        # the two sets, substantially discriminating further between the comparison and host samples.
        self.features_list = element_features_list
        self.features_list.append('Jupiter_Host')

        self.train_starname_list = self.traindata['f_preferred_name'].to_numpy()
        self.x_train = self.traindata[self.features_list].to_numpy()
        self.y_train = self.traindata['Jupiter_Host'].to_numpy()
        self.x_train_missing_i = _missing_entries_i(dataset=self.x_train)

        emptiness_train = 100*np.count_nonzero(np.isnan(self.x_train))/(np.count_nonzero(np.isnan(self.x_train))+np.count_nonzero(~np.isnan(self.x_train)))

        print('------------------------------------------------------------------------------')
        print('Dataset Emptiness Calculation:')
        print('        Training Set:', np.round(emptiness_train, 2), '% of Entries are Missing')
        print('------------------------------------------------------------------------------')
