#!/usr/bin/env python
#########################################################################
# Authors : Andrew Palmer (palmer@embl.de)
#
##########################################################################
# Version 0.1
#
# Dependencies:
# to-do
#########################################################################
import numpy as np
from pyMSpec.pyisocalc import pyisocalc
from pyMSpec.centroid_detection import gradient
ver = '0.1 (2 May. 2016)'

class Orbitrap():

    def __init__(self, resolving_power_200):
        self.resolving_power_200 = resolving_power_200

    def resolving_power_at_mz(self, mz):
        return self.resolving_power_200 * (10 / np.sqrt(mz))

    def sigma_at_mz(self, mz):
        rp = self.resolving_power_at_mz(mz)
        sigma = mz/rp/2.35482004503095 #fwhm = mz*rp, fwhm=2.3sigma
        return sigma

    def get_isotope_pattern(self,formula, adduct, charge):
        perfect_pattern = pyisocalc.perfect_pattern(pyisocalc.parseSumFormula(formula+adduct), charge=charge)
        sigma = self.sigma_at_mz(perfect_pattern.get_spectrum(source='centroids')[0][0])
        pts_per_mz = 5/sigma
        spec = pyisocalc.apply_gaussian(perfect_pattern,sigma,pts_per_mz)
        centroided_mzs, centroided_ints, _ = gradient(*spec.get_spectrum())
        spec.add_centroids(centroided_mzs, centroided_ints)
        return spec