import logging
from math import ceil
import os
import csv
from datetime import date


def _factorisation_check(datapoints, num_cnn_bins):
    """If activated, checks whether or not equipartition of dataset in desired number of bins is possible"""

    factor = False
    while factor is False:
        if datapoints % num_cnn_bins != 0:
            factors = []
            for x in range(1, 50):
                if datapoints % x == 0:
                    factors.append(x)
            print(factors)
            new_bins = int(input("Entered Number of {} CNN bins does not result in equipartitioned bins. Choose number "
                                 "of bins based on the above list of factors: \n".format(num_cnn_bins)))
            if datapoints % new_bins == 0:
                num_cnn_bins = new_bins
                factor = True
        else:
            factor = True
    return num_cnn_bins


def _spectral_order_final_binning(wl, flux, num_cnn_bins):
    """Bins data into desired number of CNN bins"""

    num_datapoints = len(wl)
    bin_size = ceil(num_datapoints / num_cnn_bins)
    final_wl, final_flux = [], []

    for bin_i in range(num_cnn_bins):
        bin_wl, bin_flux = [], []
        for bin_entry in range(bin_size):
            index = (bin_i * bin_size) + bin_entry
            if index < num_datapoints:
                bin_wl.append(wl[index]), bin_flux.append(flux[index])
            else:
                break
        final_wl.append(bin_wl), final_flux.append(bin_flux)

    return final_wl, final_flux


def _file_storage(data_destination, filename, wavlength_orders, norm_flux_orders):
    """Stores each order in a separate directory as a csv file"""

    for cnn_bin_num in range(len(wavlength_orders)):

        # Create Bin_1 Directory if it doesn't exist
        path = os.path.join(data_destination, 'Bin_{}'.format(cnn_bin_num + 1))
        if not os.path.isdir(path):
            os.mkdir(path)

        with open(os.path.join(path, '{}.csv'.format(filename)), mode='w') as csv_file:
            field_names = ['wavelength', 'normalised_flux']
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            for datapoint in range(len(wavlength_orders[cnn_bin_num])):
                writer.writerow({'wavelength': wavlength_orders[cnn_bin_num][datapoint],
                                 'normalised_flux': norm_flux_orders[cnn_bin_num][datapoint]
                                 })


def _parent_dir_gen(data_destination, custom_directory_switch, custom_directory):
    """Create Parent Directory if it doesn't exist"""
    if custom_directory_switch:
        data_destination = os.path.join(data_destination, custom_directory)
    else:
        today = date.today().strftime("%d_%m_%Y")
        data_destination = os.path.join(data_destination, 'Dataset_{}'.format(today))

    if not os.path.isdir(data_destination):
        os.mkdir(data_destination)
    return data_destination


class FinalOutputStorage:

    def __init__(self, wl, norm_flux, filename, num_cnn_bins, equipartition_bins, data_destination,
                 custom_directory_switch, custom_directory):
        """Bins Spectrum according to the desired number of CNN bins and stores them in separate directories as
        individual csv files"""

        self.wl = wl
        self.norm_flux = norm_flux
        self.filename = filename

        self.num_cnn_bins = num_cnn_bins
        self.equipartition_bins = equipartition_bins

        self.data_destination = data_destination
        self.custom_directory_switch = custom_directory_switch
        self.custom_directory = custom_directory

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='FinalOutputStorage_runtime_messages.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')

        if len(self.wl) != len(self.norm_flux):
            logging.warning('RUNTIME ERROR: Wavelength & Normalised Flux arrays for file {} have different sizes '
                            '({} & {}). Spectra Not Saved'.format(filename, len(self.wl), len(self.norm_flux)))
        else:
            if self.equipartition_bins:
                _factorisation_check(len(self.wl), self.num_cnn_bins)

            # BIN DATA INTO 'ORDERS'
            self.wavelength_orders, self.flux_orders = _spectral_order_final_binning(wl=self.wl, flux=self.norm_flux,
                                                                                     num_cnn_bins=self.num_cnn_bins)
            del self.wl, self.norm_flux

            # STORE FILES IN SEPARATE ORDER DEP DIRECTORIES
            _file_storage(data_destination=self.data_destination,
                          filename=self.filename,
                          wavlength_orders=self.wavelength_orders,
                          norm_flux_orders=self.flux_orders)
            del self.wavelength_orders, self.flux_orders
