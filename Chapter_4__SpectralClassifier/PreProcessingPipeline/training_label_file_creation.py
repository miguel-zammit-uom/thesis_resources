import csv
import os


class TrainingLabelFileCreation:
    def __init__(self, path, metadata_list):
        """Create Training Label csv"""
        self.path = path
        self.metadata_list = metadata_list
        # ['Filename', 'InvestigatorID', 'Target Name', 'Jupiter Label', 'Number of Planets']

        with open(os.path.join(path, 'training_label.csv'), mode='w') as csv_file:
            field_names = ['Filename', 'Target', 'Jupiter Host']
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            for spectrum in metadata_list:
                writer.writerow({'Filename': spectrum[0][:-5],
                                 'Target': spectrum[2],
                                 'Jupiter Host': spectrum[3]
                                 })
