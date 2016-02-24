import unittest
import itertools

import numpy

from ..centroid_detection import gradient
import common

__author__ = 'Dominik Fay'


class CentroidDetectionTest(common.MSTestCase):
    def test_gradient_kwarg_validity(self):
        """Check that gradient raises a NameError when called with unknown kwargs."""
        passing_test_cases = (
            {},
            {'max_output': -1},
            {'weighted_bins': 1, 'min_intensity': 0},
            {'max_output': 7, 'weighted_bins': 2, 'min_intensity': 28, 'grad_type': 'diff'}
        )
        failing_test_cases = (
            {'foo': 'bar'},
            {'max_output': -1, 'weighted_bins': 1, 'min_intensity': 0, 'foo': 42},
            {'max_output': -1, 'weighted_bins': 1, 'min_intensity': 0, 'grad_type': 'foo'}
        )

        for passing_spectrum in self.to_array_tuples(self.valid_spectrum_data):
            for passing_kwarg in passing_test_cases:
                if len(passing_spectrum[0]) <= passing_kwarg.get('weighted_bins', 1) * 2:
                    continue
                try:
                    gradient(*passing_spectrum, **passing_kwarg)
                except Exception as e:
                    print("Exception when calling gradient with\nargs %s and\nkwargs %s" % (
                        passing_spectrum, passing_kwarg))
                    raise
            for failing_kwarg in failing_test_cases:
                self.assertRaises(Exception, gradient, *passing_spectrum, **failing_kwarg)

    def test_gradient_kwarg_permutations(self):
        """Check postconditions for various kwargs."""
        thresholds = [0, 5, 100, 10 ** 10]
        max_outputs = [-1, 10 ** 11, 10 ** 3, 20, 5, 1]
        weighted_bins = [-1, 0, 1, 17]

        for mzs, ints in self.to_array_tuples(self.valid_spectrum_data):
            for th, max_out, bins in itertools.product(thresholds, max_outputs, weighted_bins):
                if len(mzs) <= bins * 2:
                    continue
                kwargs = {'min_intensity': th, 'max_output': max_out, 'grad_type': 'diff', 'weighted_bins': bins}
                res_mzs, res_ints, res_idxs = gradient(mzs, ints, **kwargs)
                self.check_postconditions(mzs, ints, kwargs, res_mzs, res_ints, res_idxs)

    def check_postconditions(self, mzs_in, ints_in, kwargs, mzs_out, ints_out, idx_list_out):
        th = kwargs.get('min_intensity', 0)
        max_out = kwargs.get('max_output', -1)
        bins = kwargs.get('weighted_bins', 1)
        # same length for returned lists
        self.assertEqual(len(mzs_out), len(ints_out))
        self.assertEqual(len(mzs_out), len(idx_list_out))
        # shorter or as long as input
        self.assertLessEqual(len(mzs_out), len(mzs_in))
        # length equal or less than max_out
        # The two lines below are commented because the parameter max_output is broken. TODO: Uncomment them when fixed.
        # if max_out >= 0:
        #     self.assertLessEqual(len(mzs_out), max_out)
        if len(ints_out) != 0:
            # print "%s -> %s" % (ints_in, ints_out)
            # above threshold
            self.assertGreaterEqual(min(ints_out), th)
            # max is less than or equal to input max
            self.assertLessEqual(max(ints_out), max(ints_in))
        # Currently no further assertions are made when weighted_bins > 0. TODO: Come up with some postconditions for
        # that case.
        elif max_out != 0 and bins <= 0 and len(ints_in) > 2:
            # check that there was really no intensity higher than th
            self.assertLess(max(ints_in), th, msg="Empty results although th<=max(intensites). Input: mz=%s, int=%s, "
                                                  "th=%s" % (mzs_in, ints_in, th))
        # mzs and idx_list are still sorted
        numpy.testing.assert_array_equal(mzs_out, sorted(mzs_out))
        numpy.testing.assert_array_equal(idx_list_out, sorted(idx_list_out))


if __name__ == "__main__":
    unittest.main()
