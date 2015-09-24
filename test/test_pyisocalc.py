from unittest import TestCase
from pyisocalc.pyisocalc import FormulaSegment, Element, periodic_table

__author__ = 'dominik'

import unittest


class ElementTest(TestCase):
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
        self.assertEqual(s._element, 'H')
        self.assertEqual(s._amount, 1)

    def test_raise_on_invalid_atom(self):
        invalid_atoms = ['', 'foo', 'hE']
        for a in invalid_atoms:
            self.assertRaises(ValueError, FormulaSegment, a, 1)

    def test_raise_on_invalid_number(self):
        invalid_numbers = [0, -1, 45.2]
        for n in invalid_numbers:
            self.assertRaises(ValueError, FormulaSegment, 'H', n)


if __name__ == '__main__':
    unittest.main()
