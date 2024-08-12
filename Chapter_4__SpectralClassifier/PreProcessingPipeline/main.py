from config_reader import ConfigReader
from file_prep import FileMetadataPrep
from spectra_processing import SpectrumProcessing
from final_output_storage import FinalOutputStorage, _parent_dir_gen
from training_label_file_creation import TrainingLabelFileCreation
from tqdm import tqdm
from time import sleep
import datetime

print("Initiating Startup and Loading Dataset Metadata...")
# Run ConfigReader
config_file = 'config.ini'
config_reader = ConfigReader(config_file)

data_destination = _parent_dir_gen(data_destination=config_reader.data_destination_path,
                                   custom_directory_switch=config_reader.custom_directory_switch,
                                   custom_directory=config_reader.custom_directory)

# Run FilePrep
file_metadata_prep = FileMetadataPrep(spectra_root=config_reader.spectra_root,
                                      training_label_file=config_reader.training_label_file,
                                      spectra_list_csv=config_reader.spectra_list_csv,
                                      training_label_file_index='Star')
metadata_list = file_metadata_prep.spectra_master_list
# ['Filename', 'InvestigatorID', 'Target Name', 'Jupiter Label', 'Number of Planets']
print(metadata_list)
tqdm.write("Initiating Spectral Data Calibration and Normalisation...")
sleep(0.1)
for spectrum in tqdm(metadata_list, desc="Spectral Data Calibration and Normalisation"):
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)
    # logging.basicConfig(filename='SpectrumProcessing_runtime_messages.log', filemode='w',
    #                     format='%(name)s - %(levelname)s - %(message)s')
    # Run SpectraPrep
    spectrum_process = SpectrumProcessing(spectra_root=config_reader.spectra_root,
                                          filename=spectrum[0], observer_id=spectrum[1],
                                          num_datapoints=config_reader.final_resolution,
                                          feros_elt_limits=[config_reader.lower_limit_FEROS,
                                                            config_reader.upper_limit_FEROS],
                                          max_points_to_trim=config_reader.max_points_to_trim,
                                          training_labels=spectrum[3])
    wavelength, normalised_flux = spectrum_process.wavelength, spectrum_process.normalised_flux
    if spectrum_process.error:
        with open('SpectrumProcessing_runtime_error_messages.txt', 'a') as file:
            file.write('RUNTIME ERROR {}: For {} num_trim_points {} exceeded '
                       'max_points_to_trim {}.\n'.format(datetime.datetime.now(), spectrum[0], spectrum_process.num_trim_points,
                                                         config_reader.max_points_to_trim))
    else:
        # Run OutputStorage
        final_output = FinalOutputStorage(wl=wavelength, norm_flux=normalised_flux, filename=spectrum[0],
                                          num_cnn_bins=config_reader.num_of_cnn_bins,
                                          equipartition_bins=config_reader.equipartition_bins,
                                          data_destination=data_destination,
                                          custom_directory_switch=config_reader.custom_directory_switch,
                                          custom_directory=config_reader.custom_directory)

    del wavelength, normalised_flux

if config_reader.training_labels:
    print("Generating Training Label File...")
    TrainingLabelFileCreation(path=data_destination,
                              metadata_list=metadata_list)
print("Run Completed. Dataset successfully created")
