import json
import os
import unittest
import itertools

import numpy

from pyisocalc.pyisocalc import *
from test.common import SimpleMock

__author__ = 'dominik'

element_stubs = {
    'O': SimpleMock({
        'name': lambda: 'O',
        'charge': lambda: -2,
        'number': lambda: 8,
        'masses': lambda: [15.99491462, 16.9991315, 17.9991604],
        'mass_ratios': lambda: [0.99757, 0.00038, 0.00205],
        'average_mass': lambda: 15.9994049262634
    }),
    'H': SimpleMock({
        'name': lambda: 'H',
        'charge': lambda: 1,
        'number': lambda: 1,
        'masses': lambda: [1.007825032, 2.014101778],
        'mass_ratios': lambda: [0.999885, 0.000115],
        'average_mass': lambda: 1.00794075382579
    }),
    'Fe': SimpleMock({
        'name': lambda: 'Fe',
        'charge': lambda: 3,
        'number': lambda: 26,
        'masses': lambda: [53.9396147, 55.9349418, 56.9353983, 57.9332801],
        'mass_ratios': lambda: [0.05845, 0.91754, 0.02119, 0.00282],
        'average_mass': lambda: 55.845149918245994
    })
}


class ElementTest(unittest.TestCase):
    def test_init(self):
        valid_args = ('H', 'Ar', 1, 49)
        valueerror_args = ('', 'foo', -45, 2345)
        typeerror_args = (1.0, None)
        for arg in valid_args:
            Element(arg)
        for arg in valueerror_args:
            self.assertRaises(ValueError, Element, arg)
        for arg in typeerror_args:
            self.assertRaises(TypeError, Element, arg)

    def test_probability_distribution(self):
        for s in periodic_table:
            e = Element(s)
            masses = e.masses()
            probs = e.mass_ratios()
            self.assertEquals(masses, sorted(masses))
            self.assertAlmostEqual(1.0, sum(probs), delta=0.001)
            self.assertEqual(len(masses), len(probs))


class FormulaSegmentTest(unittest.TestCase):
    def test_fields_accessible(self):
        s = FormulaSegment('H', 1)
        self.assertEqual(s.element(), 'H')
        self.assertEqual(s.amount(), 1)

    def test_raise_on_invalid_number(self):
        invalid_integers = [0, -1]
        for n in invalid_integers:
            self.assertRaises(ValueError, FormulaSegment, 'H', n)
        non_integers = [45.2, 'foo']
        for n in non_integers:
            self.assertRaises(TypeError, FormulaSegment, 'H', n)


class SumFormulaTest(unittest.TestCase):
    def test_segments_are_tuple(self):
        segments = [1, 2, 3]
        sf = SumFormula(segments)
        self.assertIsInstance(sf.get_segments(), tuple)
        self.assertSequenceEqual(segments, sf.get_segments())


class SumFormulaParserTest(unittest.TestCase):
    def test_expand_raises_on_malformed_string(self):
        syntactically_invalid_strings = ['()=', 'h2o', '']
        semantically_invalid_strings = ['ABC', 'FoO', 'Hh20']
        for s in syntactically_invalid_strings:
            self.assertRaises(ValueError, SumFormulaParser.expand, s)
        for s in semantically_invalid_strings:
            self.assertRaises(ValueError, SumFormulaParser.expand, s)

    def test_make_segments_raises_on_nonexpanded_string(self):
        syntactically_invalid_strings = ['()=', 'H((H20)3', 'h2o', '',
                                         'H(NO2)3', 'H-3', 'H0' 'H4.2']
        semantically_invalid_strings = ['ABC', 'FoO', 'Hh20']
        for s in syntactically_invalid_strings:
            self.assertRaises(ValueError, SumFormulaParser.expand, s)
        for s in semantically_invalid_strings:
            self.assertRaises(ValueError, SumFormulaParser.expand, s)

    def test_expand(self):
        test_cases = (
            ('H(NO2)3', 'HN3O6'),
            ('(N)5', 'N5'),
            ('(((H)2)3)4', 'H24'),
            ('H2O', 'H2O'),
            ('Cl(Cl2)3', 'ClCl6')
        )
        for i, o in test_cases:
            self.assertEqual(o, SumFormulaParser.expand(i))

    def test_make_segments(self):
        test_cases = (
            ('H', [FormulaSegment(Element('H'), 1)]),
            ('H2O', [FormulaSegment(Element('H'), 2), FormulaSegment(Element('O'), 1)])
        )
        for i, o in test_cases:
            self.assertSequenceEqual(o, list(SumFormulaParser.iter_segments(i)))


class SingleElementPatternTest(unittest.TestCase):
    def test_single_element_pattern(self):
        segments = (
            SegmentStub(element_stubs['H'], 1),
            SegmentStub(element_stubs['O'], 9),
            SegmentStub(element_stubs['Fe'], 348)
        )
        thresholds = (0, 1e-9, 1e-27, 1, 1e27)
        for s, th in itertools.product(segments, thresholds):
            ms = single_pattern_fft(s, th)
            nzs, ints = ms.get_spectrum()
            for i in ints:
                self.assertGreaterEqual(i, th)

    def test_raise_on_invalid_threshold(self):
        self.assertRaises(ValueError, single_pattern_fft, None, -1)


class TrimTest(unittest.TestCase):
    def test_trim(self):
        test_cases = (
            (([1], [1]), ([1.], [1.])),
            ((range(1000), [1] * 1000), ([499500.], [1.])),
            (([1, 2, 3, 4, 5, 6], [1, 2, 3, 3, 4, 5]), ([1., 2., 7., 5., 6.], [1., 2., 3., 4., 5.]))
        )
        for (i_y, i_x), (expected_y, expected_x) in test_cases:
            actual_y, actual_x = trim(i_y, i_x)
            numpy.testing.assert_array_equal(expected_x, actual_x)
            numpy.testing.assert_array_equal(expected_y, actual_y)


class IsodistTest(unittest.TestCase):
    def setUp(self):
        self.sf_stubs = {
            'H2O': SimpleMock({
                'get_segments': lambda: [SegmentStub(element_stubs['H'], 2), SegmentStub(element_stubs['O'], 1)],
                'charge': lambda: 0
            }),
            'H': SimpleMock({
                'get_segments': lambda: [SegmentStub(element_stubs['H'], 1)],
                'charge': lambda: 1
            }),
            'Fe100H89O10': SimpleMock({
                'get_segments': lambda: [SegmentStub(element_stubs['Fe'], 100), SegmentStub(element_stubs['H'], 89),
                                         SegmentStub(element_stubs['O'], 10)],
                'charge': lambda: 369
            })
        }
        self.sf_reference_values = {}
        for sf_str in self.sf_stubs:
            with open(os.path.join(os.path.dirname(__file__), 'pyisocalc_isodist_refdata_%s.json' % sf_str),
                      'r') as result_fp:
                res_dict = json.load(result_fp)
                self.sf_reference_values[sf_str] = res_dict

    def test_top_n_peaks(self):
        for sf_str in self.sf_stubs:
            sf_stub = self.sf_stubs[sf_str]
            reference_values = self.sf_reference_values[sf_str]

            actual_masses, actual_ratios = isodist(sf_stub, cutoff=0.00001).get_spectrum()
            expected_masses, expected_ratios = numpy.asarray(reference_values['mzs']), np.asarray(reference_values[
                                                                                                      'ints'])
            top10actual_masses, top10actual_ratios = self._top_n(actual_masses, actual_ratios, n=10)
            top10expected_masses, top10expected_ratios = self._top_n(expected_masses, expected_ratios, n=10)

            np.testing.assert_array_almost_equal(top10actual_masses, top10expected_masses, decimal=3)
            np.testing.assert_array_almost_equal(top10actual_ratios, top10expected_ratios, decimal=3)

    @staticmethod
    def _top_n(mzs, ints, n=10):
        indexes = numpy.argsort(ints)[::-1][:n]
        return mzs[indexes], ints[indexes]

    def test_functions(self):
        pass


class TestTranslateFwhm(unittest.TestCase):
    def test_valid_inputs(self):
        test_cases = (
            ((1, 11, 1., 2), (20, 0.42466090014400956)),
            ((492.7, 5178.3, 0.001, 25), (117140000, 0.00042466090014400956)),
        )
        for i, (expected_pts, expected_sigma) in test_cases:
            actual_pts, actual_sigma = translate_fwhm(*i)
            self.assertEqual(expected_pts, actual_pts)
            self.assertAlmostEqual(expected_sigma, actual_sigma, delta=1e-5)

    def test_raises_valueerror(self):
        invalid_inputs = (
            # min > max
            (5, 4, 1., 1),
            # <= 0 values
            (0, 0, 0., 0),
            (-3, 27, 0.01, 10),
            (3, 27, -0.01, 10),
            (3, 27, 0.01, -10)
        )
        for mi, ma, fwhm, ppfwhm in invalid_inputs:
            self.assertRaises(ValueError, translate_fwhm, mi, ma, fwhm, ppfwhm)


class TestGenGaussian(unittest.TestCase):
    def setUp(self):
        self.ms_stub = SimpleMock({'get_spectrum': lambda: (np.array([1., 2.]), np.array([50., 100.]))})

    def test_raises_errors(self):
        test_cases = (
            ((self.ms_stub, -1, 10), ValueError),
            ((self.ms_stub, 0, 10), ValueError),
            ((self.ms_stub, 10, 0), ValueError),
            ((self.ms_stub, 0, 0), ValueError),
            ((self.ms_stub, -1, 10), ValueError),
            ((self.ms_stub, 1, -3), ValueError),
            ((self.ms_stub, 1, 10.), TypeError),
        )
        for (ms, sigma, pts), e in test_cases:
            self.assertRaises(e, gen_gaussian, ms, sigma, pts)


class SegmentStub(object):
    def __init__(self, atom, number):
        self._atom = atom
        self._number = number

    def element(self):
        return self._atom

    def amount(self):
        return self._number


if __name__ == '__main__':
    unittest.main()
