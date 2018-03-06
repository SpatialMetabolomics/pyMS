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
ver = '0.2 (23 Jun. 2016)'
const_2ln2 = 2.35482004503095


class Instrument():
    def __init__(self, resolving_power, at_mz=200):
        self.resolving_power = float(resolving_power)
        self.at_mz = float(at_mz)

    def resolving_power_at_mz(self, mz):
        raise NotImplementedError

    def sigma_at_mz(self, mz):
        rp = self.resolving_power_at_mz(mz)
        sigma = mz/rp/const_2ln2 #fwhm = mz*rp, fwhm=2.3sigma
        return sigma

    def points_per_mz(self, sigma):
        return int(5 / sigma)

    def fwhm_at_mz(self, mz):
        rp = self.resolving_power_at_mz(self.at_mz)
        fwhm = mz / rp
        return fwhm

    def get_principal_peak(self, formula_adduct_string, charge):
        perfect_pattern = pyisocalc.perfect_pattern(pyisocalc.parseSumFormula(formula_adduct_string), charge=charge).get_spectrum(source='centroids')
        return perfect_pattern[0][np.argmax(perfect_pattern[1])]

    def get_isotope_pattern(self, formula_adduct_string, charge):
        perfect_pattern = pyisocalc.perfect_pattern(pyisocalc.parseSumFormula(formula_adduct_string), charge=charge)
        sigma = self.sigma_at_mz(perfect_pattern.get_spectrum(source='centroids')[0][0])
        pts_per_mz = self.points_per_mz(sigma)
        spec = pyisocalc.apply_gaussian(perfect_pattern, sigma, pts_per_mz)
        centroided_mzs, centroided_ints, _ = gradient(*spec.get_spectrum())
        spec.add_centroids(centroided_mzs, centroided_ints)
        return spec

    def generate_mz_axis(self, mz_min, mz_max, pts_per_fwhm=2):
        """
        returns mz axis
        """
        mz_axis = []
        mz = mz_min
        while mz < mz_max:
            fwhm = self.fwhm_at_mz(mz)
            step = fwhm / pts_per_fwhm
            mz_axis.append(mz + step)
            mz += step
        return np.asarray(mz_axis)


class ConstantResolvingPower(Instrument):
    def resolving_power_at_mz(self, mz):
        return self.resolving_power


class ConstantFWHM(ConstantResolvingPower):
    def sigma_at_mz(self, mz):
        rp = self.resolving_power_at_mz(self.at_mz)
        fwhm = self.at_mz/rp
        sigma = fwhm/const_2ln2
        return sigma

class Orbitrap(Instrument):
    def resolving_power_at_mz(self, mz):
        return self.resolving_power * (np.sqrt(self.at_mz) / np.sqrt(mz))


class FTICR(Instrument):
    def resolving_power_at_mz(self, mz):
        return  (self.resolving_power / mz) * self.at_mz


Orbitrap_HF = Orbitrap

TOF = ConstantResolvingPower
qTOF = TOF
Synapt = TOF
TOF_reflector = TOF