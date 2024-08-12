import os
from astropy.io import fits
import numpy as np
import warnings
import logging
from time import sleep
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum
from astropy import units as u


def _file_loader(spectra_root, filename):
    data = fits.open(os.path.join(spectra_root, filename))

    # Load wavelength and flux for the specified spectrum
    wavelength = data[1].data[0][0]
    flux = data[1].data[0][1]

    return wavelength, flux


def _threshold_check(wl):
    # Check if min, max are off beyond a threshold (hard-coded to 2Angstroms)
    min_wl, max_wl = np.min(wl), np.max(wl)
    if np.abs(3780 - int(min_wl)) > 2 or np.abs(6913 - int(max_wl)) > 2:
        logging.warning('Error: {:.3f} - {:.3f}'.format(min_wl, max_wl))


def _continuum_fit(wl, flux):
    # Fit Continuum spectrum and normalise it
    with warnings.catch_warnings(): # Ignore warnings
        warnings.simplefilter('ignore')
        spectrum = Spectrum1D(flux=flux * u.Jy, spectral_axis=wl * u.Angstrom)
        g1_fit = fit_generic_continuum(spectrum)
    cont_flux = g1_fit(np.float64(wl / 1e4) * u.um)
    return spectrum / cont_flux


class SpectrumProcessing:

    def __init__(self, spectra_root, filename, observer_id, num_datapoints, feros_elt_limits, max_points_to_trim,
                 training_labels):
        self.error = False

        self.spectra_root, self.filename = spectra_root, filename
        self.observer_id = observer_id
        self.feros_elt_limits = feros_elt_limits
        self.training_labels = training_labels
        self.num_datapoints, self.max_points_to_trim = int(num_datapoints), int(max_points_to_trim)
        self.wavelength, self.flux = _file_loader(self.spectra_root, self.filename)

        self.num_trim_points = 0

        self.wavelength, self.normalised_flux = self._spectrum_normalisation(observer_id=self.observer_id,
                                                                             wl=self.wavelength,
                                                                             flux=self.flux,
                                                                             feros_elt_limits=self.feros_elt_limits)


    def _trim_dataset(self, wl, flux, num_trim_points, max_points_to_trim, filename):
        if (num_trim_points > 1) and (num_trim_points < max_points_to_trim):
            new_wl = wl[((num_trim_points // 2) - 1):-(num_trim_points - (num_trim_points // 2) + 1)]
            new_flux = flux[((num_trim_points // 2) - 1):-(num_trim_points - (num_trim_points // 2) + 1)]
            del wl, flux
        elif (num_trim_points == 1) and (num_trim_points < max_points_to_trim):
            new_wl = wl[1:]
            new_flux = flux[1:]
        else:
            new_wl, new_flux = None, None
            self.num_trim_points = num_trim_points
            self.error = True
            # logging.warning('RUNTIME ERROR: For {} num_trim_points {} exceeded '
            #                     'max_points_to_trim {}.'.format(filename, num_trim_points, max_points_to_trim))

        return new_wl, new_flux

    def _data_binning_calibration(self, wl, flux, num_datapoints, max_points_to_trim, filename):
        if not self.error:
            bin_size = len(wl) // num_datapoints
            num_trim_points = len(wl) - (bin_size * num_datapoints)

            wl, flux = self._trim_dataset(wl, flux, num_trim_points, max_points_to_trim, filename)
            if not self.error:
                if bin_size != 1:
                    med_wl = np.ones(int(len(wl) / bin_size))
                    sum_flux = np.ones(int(len(wl) / bin_size))

                    bin_wl, bin_flux = [], []
                    for i in range(len(wl)):
                        bin_wl.append(wl[i]), bin_flux.append(flux[i])
                        if i % bin_size == 0 and i != 0:
                            med_wl[int(i / 3) - 1] = np.median(bin_wl)
                            sum_flux[int(i / 3) - 1] = np.sum(bin_flux)

                            bin_wl, bin_flux = [], []
                    med_wl[-1] = np.median(bin_wl)
                    sum_flux[-1] = np.sum(bin_flux)

                    return med_wl, sum_flux

                else:
                    return wl, flux
            else:
                return None, None
        else:
            return None, None

    def _spectrum_normalisation(self, observer_id, wl, flux, feros_elt_limits):
        if observer_id == 'Neves' or observer_id =='HARPS':
            if not self.error:
                wl, flux = self._data_binning_calibration(wl=wl, flux=flux,
                                                          num_datapoints=self.num_datapoints,
                                                          max_points_to_trim=self.max_points_to_trim,
                                                          filename=self.filename)
                if not self.error:
                    _threshold_check(wl)

                    norm_spectrum = _continuum_fit(wl=wl, flux=flux)
                    return norm_spectrum.spectral_axis, norm_spectrum.flux
                else:
                    return None, None
            else:
                return None, None
        elif observer_id == 'Bodaghee' or observer_id == 'Ecuvillon' or observer_id =='FEROS':
            if not self.error:
                wl, flux = self._data_binning_calibration(wl=wl[feros_elt_limits[0]:feros_elt_limits[1]],
                                                          flux=flux[feros_elt_limits[0]:feros_elt_limits[1]],
                                                          num_datapoints=self.num_datapoints,
                                                          max_points_to_trim=self.max_points_to_trim,
                                                          filename=self.filename)

                if not self.error:
                    _threshold_check(wl)

                    norm_spectrum = _continuum_fit(wl=wl, flux=flux)
                    return norm_spectrum.spectral_axis, norm_spectrum.flux
                else:
                    return None, None
            else:
                return None, None
        else:
            self.error = True
            return None, None
