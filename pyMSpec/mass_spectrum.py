import numpy as np

import pyMSpec
class MassSpectrum:
    """
    a data container for a single mass spectrum
    includes methods for signal processing
    """
    def __init__(self, profile_spec=[], centroid_spec=[]):
        self._mzs = []
        self._intensities = []
        self._centroids = []
        self._centroids_intensity = []
        self._processing = []
        if profile_spec != []:
            self._mzs, self._intensities = profile_spec
        if centroid_spec != []:
            self._centroids, self._centroids_intensity = centroid_spec

    # Private basic spectrum I/O
    def __add_mzs(self, mzs):
        self._mzs = mzs

    def __add_intensities(self, intensities):
        self._intensities = intensities

    def __get_mzs(self):
        return np.asarray(self._mzs)

    def __get_intensities(self):
        return np.asarray(self._intensities)

    def __get_mzs_centroids(self):
        return np.asarray(self._centroids)

    def __get_intensities_centroids(self):
        return np.asarray(self._centroids_intensity)

    def __add_centroids_mzs(self, mz_list):
        self._centroids = mz_list

    def __add_centroids_intensities(self, intensity_list):
        self._centroids_intensity = intensity_list

    # Public methods
    def add_spectrum(self, mzs, intensities):
        if len(mzs) != len(intensities):
            raise IOError("mz/intensities vector different lengths")
        self.__add_mzs(mzs)
        self.__add_intensities(intensities)

    def add_centroids(self, mz_list, intensity_list):
        if len(mz_list) != len(intensity_list):
            raise IOError("mz/intensities vector different lengths")
        self.__add_centroids_mzs(mz_list)
        self.__add_centroids_intensities(intensity_list)

    def get_spectrum(self, source='profile'):
        if source == 'profile':
            mzs = self.__get_mzs()
            intensities = self.__get_intensities()
        elif source == 'centroids':
            mzs = self.__get_mzs_centroids()
            intensities = self.__get_intensities_centroids()
        else:
            raise IOError('spectrum source should be profile or centroids')
        return mzs, intensities

    def normalise_spectrum(self, method="tic", method_args={}):
        from pyMSpec import normalisation
        self._centroids_intensity = normalisation.apply_normalisation(self._centroids, self._centroids_intensity, method, method_args)
        self._intensities = normalisation.apply_normalisation(self._mzs, self._intensities, method, method_args)
        self._processing.append(method)
        return self

    def smooth_spectrum(self, method="sg_smooth", method_args={}):
        from pyMSpec import smoothing
        self._mzs, self._intensities = smoothing.apply_smoothing(self._mzs, self._intensities, method, method_args)
        self._processing.append(method)
        return self

# for compatibility with andy-d-palmer/pyIMS
mass_spectrum = MassSpectrum

class MSn_spectrum(MassSpectrum):
    """
    a data container for fragmentation spectrum
    """
    def __init__(self, ms_level=""):
        self.ms_transitions = []
        self.ms_level = ms_level

    def add_transition(self, transitions):
        # transitions is a list of ms fragmentation acceptance windows
        self.transitions = transitions
        self.mz_level = len(self.transitions) + 1
