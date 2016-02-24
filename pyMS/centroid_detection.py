import numpy as np


def gradient(mzs, intensities, **opt_args):
    function_args = {'max_output': -1, 'weighted_bins': 1, 'min_intensity': 1e-5, 'grad_type': 'gradient'}
    for key, val in opt_args.iteritems():
        if key in function_args.keys():
            function_args[key] = val
        else:
            print 'possible arguments:'
            for i in function_args.keys():
                print i
            raise NameError('gradient does not take argument: %s' % key)
    # TODO: temporary workaround to disable the parameter until it is fixed.
    mzMaxNum = -1
    weighted_bins = function_args['weighted_bins']
    min_intensity = function_args['min_intensity']
    gradient_type = function_args['grad_type']
    assert mzMaxNum < len(mzs)
    assert len(mzs) == len(intensities)
    assert weighted_bins < len(mzs) / 2.
    # calc first&sectond differential
    if gradient_type == 'gradient':
        MZgrad = np.gradient(intensities)
        MZgrad2 = np.gradient(MZgrad)[0:-1]
    elif gradient_type == 'diff':
        MZgrad = np.concatenate((np.diff(intensities), [1]))
        MZgrad2 = np.diff(MZgrad)
    else:
        raise ValueError('gradient type {} not known'.format(gradient_type))
    # detect crossing points
    cPoint = MZgrad[0:-1] * MZgrad[1:] <= 0
    mPoint = MZgrad2 < 0
    indices = cPoint & mPoint
    # bool->list of indices
    # Could check left/right of crossing point
    indices_list_l = np.where(cPoint & mPoint)[0]
    indices_list_r = indices_list_l + 1
    indices_list = np.where(
        intensities[indices_list_l] > intensities[indices_list_r],
        indices_list_l, indices_list_r)
    indices_list = np.unique(indices_list)

    # Remove any 'peaks' that aren't real
    indices_list = indices_list[intensities[indices_list] > min_intensity]

    # Select the peaks
    intensities_list = intensities[indices_list]
    mzs_list = mzs[indices_list]

    # Tidy up if required
    if mzMaxNum > 0:
        if len(mzs_list) > mzMaxNum:
            sort_idx = np.argsort(intensities_list)
            intensities_list = intensities_list[sort_idx[-mzMaxNum:]]
            mzs_list = mzs_list[sort_idx[-mzMaxNum:]]
            indices_list = indices_list[sort_idx[-mzMaxNum:]]
        elif len(mzs) < mzMaxNum:
            # FIXME: for what purpose are we appending zeros here?
            lengthDiff = mzMaxNum - len(indices_list)
            pad = np.zeros((lengthDiff, 1))
            mzs_list = np.concatenate((mzs_list, pad))
            intensities_list = np.concatenate((intensities_list, pad))
            indices_list = np.concatenate((indices_list, pad))

    if weighted_bins > 0:
        # check no peaks within bin width of spectrum edge
        good_idx = (indices_list > weighted_bins) & (
            indices_list < (len(mzs) - weighted_bins))
        mzs_list = mzs_list[good_idx]
        intensities_list = intensities_list[good_idx]
        indices_list = indices_list[good_idx]
        r = pick_max_(mzs, intensities, mzs_list, intensities_list,
                      indices_list, weighted_bins)
        mzs_list = r[0, :]
        intensities_list = r[1, :]
        indices_list = r[2, :].astype(int)
    return (mzs_list, intensities_list, indices_list)


def pick_max_(mzs, intensities, mzs_list, intensities_list, indices_list,
              weighted_bins):
    result = np.zeros((3, len(mzs_list)))
    for ii in xrange(len(mzs_list)):
        s = w = 0.0
        max_intensity_idx = 0
        max_intensity = -1
        for k in xrange(-weighted_bins, weighted_bins + 1):
            idx = indices_list[ii] + k
            mz = mzs[idx]
            intensity = intensities[idx]
            w += intensity
            s += mz * intensity
            if intensity > max_intensity:
                max_intensity = intensity
                max_intensity_idx = idx
        result[0][ii] = s / w
        result[1][ii] = max_intensity
        result[2][ii] = max_intensity_idx
    return result


try:
    from numba import njit
    pick_max_ = njit(pick_max_)
except ImportError:
    pass
