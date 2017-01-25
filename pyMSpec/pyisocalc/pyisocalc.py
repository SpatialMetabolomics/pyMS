#!/usr/bin/env python
#########################################################################
# Authors : Andrew Palmer (palmer@embl.de)
#           Artem Tarasov (lomereiter@gmail.com)
#           Dominik Fay (dominik.fay@embl.de)
# Modified from v0.8 of code by: Andy Ohlin (debian.user@gmx.com)
#
##########################################################################
# Version 0.3
#
# Dependencies:
# python2.7, python-numpy
#########################################################################
import functools

import numpy as np
from numpy import exp  # misc math functions
from numpy import asarray  # for cartesian product
from six import iteritems
from six.moves import xrange

from ..mass_spectrum import MassSpectrum
from ..centroid_detection import gradient
from .periodic_table import periodic_table
from .canopy.sum_formula import parse as canopy_sum_formula_parse
from .canopy.sum_formula_actions import Actions as SumFormulaActions
from .canopy.sum_formula_actions import InvalidFormulaError
from .canopy.sum_formula import ParseError

ver = '0.3 (4 Jan. 2016)'

mass_electron = 0.00054857990924

element_order = dict((e, k) for k, e in enumerate(periodic_table.keys()))

@functools.total_ordering
class Element(object):
    """
    An element from the periodic table.
    """

    def __init__(self, id):
        """
        Initializes an Element given some identifier
        :param id: Can be either the element symbol as a string,
        e.g. 'H', 'C', 'Ag' or its atomic number in the periodic table
        """
        if isinstance(id, str):
            try:
                data = tuple(periodic_table[id])
                self._initialize(id, *data)
                # self._name = id
                # self._number = data[0]
                # self._masses = data[1]
                # self._ratios = data[2]
            except KeyError:
                raise ValueError("%s is not an element." % id)
        elif isinstance(id, int):
            if id <= 0:
                raise ValueError("%s is not greater than zero." % id)
            if id >= len(periodic_table):
                raise ValueError("There is no element with atomic number %s" % id)
            periodic_table_list = list(periodic_table.items())
            s, data = tuple(periodic_table_list[id])
            self._initialize(s, *data)
        else:
            raise TypeError("id must be either str or int, not " % id.__class__)

    def name(self):
        """
        Return the element symbol as a string.
        """
        return self._name

    def charge(self):
        """
        Return the atomic charge of this element.
        :rtype: int
        """
        return self._charge

    def number(self):
        """
        Return the atomic number of this element in the periodic table.
        :rtype: int
        """
        return self._number

    def masses(self):
        """
        Return the masses of all possible isotopes in ascending order.

        :rtype: sequence
        """
        return self._masses

    def mass_ratios(self):
        """
        Return the probability of each isotope, ordered by the isotope's
        atomic mass.
        :rtype: sequence
        """
        return self._ratios

    def average_mass(self):
        """
        Return the average mass of this element, that is the dot product of its masses and ratios.
        :rtype: float
        """
        return np.dot(self._masses, self._ratios)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._name == other.name() and self._number == other.number()
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return element_order[self._name] < element_order[other._name]
        else:
            return NotImplemented

    def __str__(self):
        return self._name

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__class__.__name__ + "(id='%s')" % self.name()

    def _initialize(self, name, number, charge, masses, ratios):
        self._name = name
        self._number = number
        self._charge = charge
        self._masses = masses
        self._ratios = ratios

@functools.total_ordering
class FormulaSegment(object):
    """
    A segment from an expanded molecular sum formula.

    A segment is a single element together with its number of occurrences
    within a molecule. For example, the molecule 'H20' would consist of the
    segments ('H', 2) and ('O', 1).
    """

    def __init__(self, element, amount):
        """
        Create a segment given an element and a number.
        :param element: an Element object
        :param amount: a positive non-zero integer
        """
        if not isinstance(amount, int):
            raise TypeError("%s is not an integer." % amount)
        if not amount > 0:
            raise ValueError("number must be greater than 0, but is %s" % amount)
        self._element = element
        self._amount = amount

    def element(self):
        """
        Return the chemical element.
        """
        return self._element

    def amount(self):
        """
        Return the amount of the element.
        """
        return self._amount

    def charge(self):
        """
        The element's charge multiplied by its amount.
        """
        return self._element.charge() * self._amount

    def average_mass(self):
        """
        The element's average mass multiplied by its amount.
        """
        return self._element.average_mass() * self._amount

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._element == other.element() and self._amount == other.amount()
        else:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self._element == other._element:
            return self._amount < other._amount
        return self._element < other._element

    def __str__(self):
        if self._amount > 1:
            return "%s%s" % (self._element, self._amount)
        else:
            return str(self._element)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return "FormulaSegment(element=%s, number=%s)" % (repr(self._element),
                                                          self._amount)


class SumFormula(object):
    """
    A molecular sum formula, built up from FormulaSegments. Use parseSumFormula
    to parse your string representation into a SumFormula object.

    To get the expanded string representation of this object, use str().
    """

    def __init__(self, segments):
        """
        :param segments: sequence of FormulaSegments
        """
        self._segments = tuple(segments)

    def get_segments(self):
        """
        Return the sequence of segments of which this sum formula consists.
        :rtype: tuple
        """
        return self._segments

    def average_mass(self):
        """
        The sum of the average masses of its segments.
        :rtype: float
        """
        return sum(map(lambda x: x.average_mass(), self._segments))

    def charge(self):
        """
        The sum of the charges of its segments.
        :rtype: int
        """
        return sum(map(lambda x: x.charge(), self._segments))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._segments == other.get_segments()
        else:
            return NotImplemented

    def __str__(self):
        return ''.join(map(str, self._segments))

    def __unicode__(self):
        return self.__str__()

def parseSumFormula(string):
    """
    Parses string representation of a sum formula into a list of FormulaSegments
    """
    actions = SumFormulaActions()
    counts = canopy_sum_formula_parse(string, actions)
    return SumFormula([FormulaSegment(Element(str(k)), v) for k, v in iteritems(counts)])

def single_pattern_fft(segment, threshold=1e-9):
    """
    .. py:function:: single_pattern_fft(segment, [threshold=1e-9])

    Calculates the isotope pattern of a single FormulaSegment using multidimensional fast fourier transform.

    See 'Efficient Calculation of Exact Fine Structure Isotope Patterns via the
    Multidimensional Fourier Transform' (A. Ipsen, 2014).

    :type segment: FormulaSegment
    :param threshold: Only intensities above this threshold will be part of the result. Must be a non-negative number
    :type threshold: float
    :return: the isotopic pattern as a MassSpectrum
    :rtype: MassSpectrum
    """
    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    element, amount = segment.element(), segment.amount()
    iso_mass, iso_abundance = np.asarray(element.masses()), np.asarray(element.mass_ratios())
    res = MassSpectrum()
    if len(iso_abundance) == 1:
        res.add_spectrum(iso_mass * amount, np.array([1.0]))
        return res
    if amount == 1:
        significant = np.where(iso_abundance > threshold)
        res.add_spectrum(iso_mass[significant], iso_abundance[significant])
        return res
    dim = len(iso_abundance) - 1
    if (amount + 1) ** dim > 1e7:
        raise Exception("Can't compute isotope pattern for" + str(segment))
    abundance = np.zeros([amount + 1] * dim)
    abundance.flat[0] = iso_abundance[0]
    abundance.flat[(amount + 1) ** np.arange(dim)] = iso_abundance[-1:0:-1]
    abundance = np.real(np.fft.ifftn(np.fft.fftn(abundance) ** amount))
    significant = np.where(abundance > threshold)
    intensities = abundance[significant]
    masses = amount * iso_mass[0] + (iso_mass[1:] - iso_mass[0]).dot(significant)
    res.add_spectrum(masses, intensities)
    return res


def trim(y, x):
    """
    .. py:function:: trim(y, x)

    Remove duplicate elements in the second array and sum the duplicate values in the first one.
    This function returns an array containing the unique values from y and an array containing the values from x
    where its elements at the indexes of the duplicate elements in y have been summed.

    Example:

    .. code::
    >>> trim([5, 1, 2, 5], [1, 2, 2, 3])
    (array([ 5.,  3.,  5.]), array([1, 2, 3]))

    :param y: the array from which the values are summed
    :type y: Union[ndarray, Iterable]
    :param x: the array from which the duplicates are removed
    :type x: Union[ndarray, Iterable]
    :return: the trimmed y array and the trimmed x array
    :rtype: Tuple[ndarray]
    """
    x, inv = np.unique(x, return_inverse=True)
    y = np.bincount(inv, weights=y)
    return y, x


def cartesian(rx, mx, threshold=0.0001):
    """
    .. py:function:: cartesian(rx, mx, threshold)

    Combine multiple isotope patterns into a single one.

    :param rx: Sequence of ratio arrays
    :type rx: Sequence[ndarray]
    :param mx: Sequence of mass arrays
    :type mx: Sequence[ndarray]
    :param threshold: threshold below which the resulting ratios are filtered out
    :return: The resulting ratio array and the mass array
    :rtype: Tuple[ndarray]
    """
    ry, my = asarray(rx[0]), asarray(mx[0])
    for i in xrange(1, len(rx)):
        newr = np.outer(rx[i], ry).ravel()
        newm = np.add.outer(mx[i], my).ravel()
        js = np.where(newr > threshold)[0]
        ry, my = newr[js], newm[js]
    return ry, my


##################################################################################
# Does housekeeping to generate final intensity ratios and puts it into a dictionary
##################################################################################
def normalize(m, n, charges, cutoff):
    m, n = np.asarray(m).round(8), np.asarray(n).round(8)
    n *= 100.0 / max(n)
    filter = n > cutoff
    m, n = m[filter], n[filter]
    if charges != 0:
        m -= charges * mass_electron
        m /= abs(charges)
    return m, n


def gen_gaussian(ms, sigma, pts):
    """
    Transform each peak in an isotope pattern into a gaussian curve.

    Each of the curves is scaled up to the intensity value for the corresponding m/z. The output will be on a regular
    grid with pts points, starting from min_mz - 1, up to max_mz + 1, where min_mz is the lowest m/z value and max_mz is
    the highest m/z value. Since each curve is rendered on the same grid, overlapping curves will add up.

    :param ms: the isotope pattern as a MassSpectrum object
    :return: the smoothed pattern
    :rtype: Tuple[ndarray]
    :throws ValueError: if sigma or pts are not greater than 0
    :throws TypeError: if pts is not an integer
    """
    if not isinstance(pts, int):
        raise TypeError("pts must be an integer")
    if min(sigma, pts) <= 0:
        raise ValueError("sigma and pts must be greater than 0")
    mzs, intensities = ms.get_spectrum(source="centroids")
    xvector = np.linspace(min(mzs) - 1, max(mzs) + 1, pts)
    if len(mzs) * len(xvector) < 1e6:
        yvector = intensities.dot(exp(-0.5 * (np.add.outer(mzs, -xvector) / sigma) ** 2))
    else:
        yvector = np.zeros_like(xvector)
        for mz, intensity in zip(mzs, intensities):
            yvector += intensity * exp(-0.5 * (np.add.outer(mz, -xvector) / sigma) ** 2)
    return xvector, yvector


def gen_approx_gaussian(ms, sigma, pts, n=20):
    """
    Approximate and faster version of gen_gaussian

    :param ms: the isotope pattern as a MassSpectrum object
    :return: the smoothed pattern
    :rtype: Tuple[ndarray]
    :throws ValueError: if sigma or pts are not greater than 0
    :throws TypeError: if pts is not an integer
    """
    if not isinstance(pts, int):
        raise TypeError("pts must be an integer")
    if min(sigma, pts) <= 0:
        raise ValueError("sigma and pts must be greater than 0")
    mzs, intensities = ms.get_spectrum()
    mzs = mzs / sigma
    xvector = np.linspace(min(mzs) - 1.0/sigma, max(mzs) + 1.0/sigma, pts)
    yvector = np.zeros(xvector.shape)
    for mz, intensity in zip(mzs, intensities):
        k = xvector.searchsorted(mz)
        l = max(k - n, 0)
        r = min(k + n + 1, len(xvector))
        yvector[l:r] += intensity * np.exp(-0.5 * (mz - xvector[l:r]) ** 2)
    return xvector * sigma, yvector


def total_points(min_x, max_x, points_per_mz):
    """
    Calculate the number of points for the regular grid based on the full width at half maximum.

    :param min_x: the lowest m/z value
    :param max_x: the highest m/z value
    :param points_per_mz: number of points per fwhm
    :return: total number of points
    :rtype: int
    """
    if min_x > max_x:
        raise ValueError("min_x > max_x")
    if min(min_x, max_x, points_per_mz) <= 0:
        raise ValueError("all inputs must be greater than 0")
    return int((max_x - min_x) * points_per_mz) + 1


def fwhm_to_sigma(min_x, max_x, fwhm):
    """
    When fitting an isotope pattern to a gaussian distribution, this function calculates the standard deviation sigma
    for the gaussian distribution.

    :param min_x: the lowest m/z value
    :param max_x: the highest m/z value
    :param fwhm: the full width at half maximum
    :return: Sigma
    :rtype: float
    :throws ValueError: if min_x > max_x or if not all inputs are greater than 0
    """
    if min_x > max_x:
        raise ValueError("min_x > max_x")
    if min(min_x, max_x, fwhm) <= 0:
        raise ValueError("all inputs must be greater than 0")
    sigma = fwhm / 2.3548200450309493  # approximation of 2*sqrt(2*ln2)
    return sigma


########
# main function#
########
def complete_isodist(sf, sigma=0.001, cutoff_perc=0.1, charge=None, pts_per_mz=10000, centroid_func=gradient,
                     centroid_kwargs=None):
    """
    Wrapper function for applying perfect_pattern, then gen_gaussian and eventually centroid detection.

    :param sf: the sum formula
    :param sigma: Full width at half maximum
    :type sigma: float
    :param cutoff_perc: min percentage of the maximum intensity to return, max value = 100
    :type cutoff_perc: float
    :param charge: charge of the molecule
    :type charge: int
    :param pts_per_mz: Number of points per mz for the regular grid
    :param centroid_func: the centroid function to apply to the isotope pattern or None if no centroid detection
    should be performed. Must have the same signature as centroid_detection.gradient.
    :param centroid_kwargs: dict to pass to centroid_func as optional parameters
    :return:
    """
    ms1 = perfect_pattern(sf, cutoff_perc, charge=charge)
    ms2 = apply_gaussian(ms1, sigma, pts_per_mz)
    if centroid_func:
        centroid_kwargs = centroid_kwargs or {'weighted_bins': 5}
        centroid_kwargs['min_intensity'] = cutoff_perc
        centroided_mzs, centroided_ints, _ = centroid_func(*ms2.get_spectrum(), **centroid_kwargs)
        ms2.add_centroids(centroided_mzs, centroided_ints)
    return ms2


def perfect_pattern(sf, cutoff_perc=0.1, single_pattern_func=single_pattern_fft, charge=None):
    """
    Compute the isotope pattern of a molecule given by its sum formula.

    First applies single_pattern_func to each segment within the sum formula, then combines these individual patterns
    into a single one.

    :param sf: the sum formula
    :type sf: SumFormula
    :param cutoff_perc: min percentage of the maximum intensity to return, max value = 100
    :type cutoff_perc: float
    :param single_pattern_func: the function to compute a single isotope pattern. Must have the same signature as
    single_pattern_fft
    :param charge: charge of the molecule
    :type charge: int
    :return: the combined isotope pattern as a mass spectrum
    :rtype: MassSpectrum
    """
    single_patterns = (single_pattern_func(segment) for segment in sf.get_segments())
    pattern_list = list(p.get_spectrum() for p in single_patterns)
    single_pattern_masses, single_pattern_ratios = zip(*pattern_list)
    combined_ratios, combined_masses = trim(*cartesian(single_pattern_ratios, single_pattern_masses))
    # intensity_filter = combined_ratios > cutoff_perc
    # combined_ratios, combined_masses = combined_ratios[intensity_filter], combined_masses[intensity_filter]
    if charge is None:
        charge = sf.charge()
    normalized_masses, normalized_ratios = normalize(combined_masses, combined_ratios, charge, cutoff_perc)
    ms = MassSpectrum()
    ms.add_centroids(normalized_masses, normalized_ratios)
    return ms


def apply_gaussian(ms_input, sigma, pts_per_mz=10, exact=True):
    """
    Smooth every peak into a gaussian shape using instrument-specific configuration.

    :param ms_input: the mass spectrum
    :type ms_input: MassSpectrum
    :param sigma: sigma parameter for Gaussian. See fwhm_to_sigma
    :type fwhm: float
    :param pts_per_mz: Number of points per one mz unit for the regular grid
    :type pts_per_mz: int
    :param exact: if False, this function may use an approximate implementation
    :return: a new mass spectrum containing the smoothed data in both profile and centroid mode
    """
    if min(sigma, pts_per_mz) <= 0:
        raise ValueError("sigma and pts_per_mz must be greater than 0")
    input_mzs, input_ints = ms_input.get_spectrum(source="centroids")
    ms_output = MassSpectrum()
    pts = total_points(min(input_mzs) - 1, max(input_mzs) + 1, pts_per_mz)
    if exact:
        gauss_mzs, gauss_ints = gen_gaussian(ms_input, sigma, pts)
    else:
        gauss_mzs, gauss_ints = gen_approx_gaussian(ms_input, sigma, pts)
    gauss_ints *= 100.0 / max(gauss_ints)
    ms_output.add_spectrum(gauss_mzs, gauss_ints)
    return ms_output
