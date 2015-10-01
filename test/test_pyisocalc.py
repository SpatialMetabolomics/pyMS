import unittest
import itertools

from pyisocalc.pyisocalc import *

__author__ = 'dominik'


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
            SegmentStub(HStub(), 1),
            SegmentStub(OStub(), 9),
            SegmentStub(FeStub(), 348)
        )
        thresholds = (0, 1e-9, 1e-27, 1, 1e27)
        for s, th in itertools.product(segments, thresholds):
            ms = single_element_pattern(s, th)
            nzs, ints = ms.get_spectrum()
            for i in ints:
                self.assertGreaterEqual(i, th)

    def test_raise_on_invalid_threshold(self):
        self.assertRaises(ValueError, single_element_pattern, None, -1)


class SegmentStub(object):
    def __init__(self, atom, number):
        self._atom = atom
        self._number = number

    def element(self):
        return self._atom

    def amount(self):
        return self._number


class ElementStub(object):
    def __getattribute__(self, name):
        return object.__getattribute__(self, '_data')[name]


class HStub(ElementStub):
    def __init__(self):
        self._data = {
            'name': lambda: 'H',
            'charge': lambda: 1,
            'number': lambda: 1,
            'masses': lambda: [1.007825032, 2.014101778],
            'mass_ratios': lambda: [0.999885, 0.000115],
            'average_mass': lambda: 1.00794075382579
        }


class OStub(ElementStub):
    def __init__(self):
        self._data = {
            'name': lambda: 'O',
            'charge': lambda: -2,
            'number': lambda: 8,
            'masses': lambda: [15.99491462, 16.9991315, 17.9991604],
            'mass_ratios': lambda: [0.99757, 0.00038, 0.00205],
            'average_mass': lambda: 15.9994049262634
        }


class FeStub(ElementStub):
    def __init__(self):
        self._data = {
            'name': lambda: 'Fe',
            'charge': lambda: 3,
            'number': lambda: 26,
            'masses': lambda: [53.9396147, 55.9349418, 56.9353983, 57.9332801],
            'mass_ratios': lambda: [0.05845, 0.91754, 0.02119, 0.00282],
            'average_mass': lambda: 55.845149918245994
        }


if __name__ == '__main__':
    unittest.main()
