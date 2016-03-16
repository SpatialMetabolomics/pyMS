__author__ = 'palmer'
import numpy as np
def apply_smoothing(mzs, counts, type_str="", method_args={}):
    """
    helper function to apply a smoothing function (with some input testing etc)
    :param counts: numpy array of values to smooth
    :param type_str: smooting type to apply (name)
    :return: tuple (mzs: numpy array, counts: numpy array)
    """
    smoothToApply = {"none": nosmooth,
                   "sg_smooth": sg_smooth,
                   "apodization": apodization,
                   "rebin": rebin,
                   "fast_change": fast_change,
                   "median":median
                   }
    if type_str not in smoothToApply.keys():
        raise ValueError("{} not in {}".format(type_str, smoothToApply.keys()))
    np.testing.assert_array_almost_equal(len(mzs), len(counts))
    mzs = np.asarray(mzs, dtype=float)
    counts = np.asarray(counts, dtype=float)
    return smoothToApply[type_str](mzs,counts, **method_args)



def nosmooth(mzs, intensities):
    """
    does nothing, just returns input. is a dummy for programmatic case where a function must be supplied
    :param counts:  numpy array
    :return: mzs: numpy array
    :return: counts: numpy array
    """
    return mzs, intensities


def sg_smooth(mzs, intensities, n_smooth=1, w_size=5):
    """
    sav gol
    :param mzs:  numpy array numpy array of mz values
    :param counts:  numpy array numpy array of values to smooth
    :param n_smooth:  int number of times to apply smoothing
    :param w_size:  int window size
    :return: mzs: numpy array
    :return: counts: numpy array
    """
    import scipy.signal as signal
    for n in range(0, n_smooth):
        intensities = signal.savgol_filter(intensities, w_size, 2)
    intensities[intensities < 0] = 0
    return mzs, intensities


def apodization(mzs, intensities, w_size=10):
    """
    apodization with slepian window
    :param mzs:  numpy array numpy array of mz values
    :param counts:  numpy array numpy array of values to smooth
    :param w_size:  int window size
    :return: mzs: numpy array
    :return: counts: numpy array
    """
    import scipy.signal as signal
    win = signal.hann(w_size)
    win = signal.slepian(w_size, 0.3)
    intensities = signal.fftconvolve(intensities, win, mode='same') / sum(win)
    intensities[intensities < 1e-6] = 0
    return mzs, intensities


def rebin(mzs, intensities, delta_mz=0.1):
    """
    rebin spectrum
    :param mzs:  numpy array numpy array of mz values
    :param counts:  numpy array numpy array of values to smooth
    :param delta_mz:  float, new mz bin width (constant across mz axis)
    :return: mzs: numpy array
    :return: counts: numpy array
    """
    import numpy as np
    n_bins = np.round((mzs[-1] - mzs[0]) / delta_mz)
    new_mzs = np.linspace(mzs[0], mzs[-1] + delta_mz, n_bins)
    mz_idx = np.digitize(mzs, new_mzs[0:-1])
    new_intensities = np.bincount(mz_idx, weights=intensities, minlength=len(new_mzs))
    return new_mzs, new_intensities


def fast_change(mzs, intensities, diff_thresh=0.01):
    """
    remove high frequency noise from the data
    :param mzs:  numpy array numpy array of mz values
    :param counts:  numpy array numpy array of values to smooth
    :param diff_thresh:  float numeric change to remove
    :return: mzs: numpy array
    :return: counts: numpy array
    """
    import numpy as np
    import scipy.signal as signal
    diff = np.concatenate((np.abs(np.diff(intensities)), [1]))
    diff = signal.medfilt(diff)
    intensities[diff < diff_thresh] = 0
    return mzs, intensities


def median(mzs, intensities, w_size=3):
    """
    apply median filter
    :param mzs:  numpy array numpy array of mz values
    :param counts:  numpy array numpy array of values to smooth
    :param w_size:  int window size
    :return: mzs: numpy array
    :return: counts: numpy array
    """
    import scipy.signal as signal
    return mzs, signal.medfilt(intensities, kernel_size=w_size)
