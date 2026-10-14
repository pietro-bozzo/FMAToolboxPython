""" Utilities to handle time intervals for FMAToolbox """

import numpy as np


def consolidate(intervals, eps:float=None, duration:float=None):
    """remove overlaps in a set of intervals, yielding its most compact description (the union of its elements)
    e.g., [[1,4],[2,6]] will become [[1,6]]

    arguments:
        intervals    (:,2) float, rows are [start, stop] times for an interval (s); if 1d, reshaped to (1,2)
        eps          float = 0, intervals with bounds closer than eps are also consolidated
        duration     float = 0, trim consolidated set of intervals so that it has given total duration,
                     if negative, duration is counted from the end and intervals are trimmed rightwards

    output:
        intervals    (:,2) float, consolidated intervals (s)
    """

    intervals = np.array(intervals,dtype=float,ndmin=2)
    if intervals.shape[1] != 2:
        raise ValueError("'intervals' must be a (n,2) array")
    if (intervals[:,0] > intervals[:,1]).any():
        raise ValueError("rows of 'intervals' must be increasing")
    if eps is None: eps = 0.
    if duration is None: duration = 0.

    # remove [nan nan] rows, handle empty input
    intervals = intervals[~np.all(np.isnan(intervals),axis=1)]
    if intervals.size == 0:
        return intervals

    # widen intervals
    if eps:
        intervals = intervals + np.array([-1,1]) * eps

    # sort by start time
    intervals = intervals[intervals[:,0].argsort()]

    # flatten and argsort to find overlaps
    intervals = intervals.flatten()
    ind = intervals.argsort()

    # remove all ind which are followed by at least one smaller element
    m = ind[-2:].min()
    is_ok = [True,ind[-2] < ind[-1]]
    for i in range(3,ind.shape[0] + 1):
        is_ok.append(ind[-i] < m)
        m = min(ind[-i],m)
    is_ok.reverse()
    ind = ind[is_ok]

    # remove consecutive odd elements
    is_odd = (ind % 2).astype(bool)
    ind = ind[np.concatenate((~is_odd[:-1] | ~is_odd[1:],[True]))]

    # rebuild intervals
    intervals = intervals[np.reshape(ind,(ind.shape[0] // 2,2))]

    # re-shorten intervals
    if eps:
        intervals = intervals + np.array([1,-1]) * eps

    # trim
    if duration != 0:
        intervals = trim(intervals,duration,fast=True,eps=eps) # fast = True avoids infinite calls

    return intervals


def trim(intervals, duration:float, fast:bool=None, eps:float=None):
    """trim intervals so that it has given total duration

    arguments:
        intervals    (:,2) float, rows are [start, stop] times for an interval (s); if 1d, reshaped to (1,2)
        duration     float, maximum total durationt to keep, if negative, duration is counted from the end and
                     'intervals' are trimmed rightwards
        fast         bool = False, if True, 'intervals' are expected to be consolidated (see `consolidate`)
        eps          float = 0, intervals with bounds closer than eps are also consolidated (only for False 'fast')

    output:
        intervals    (:,2) float, trimmed intervals (s)
    """

    if not fast:
        intervals = consolidate(intervals,eps=eps)
    if intervals.size == 0:
        return intervals

    if duration > 0:
        cum_duration = np.cumsum(np.diff(intervals))
        if cum_duration[-1] > duration:
            idx = np.searchsorted(cum_duration,duration) + 1 # first interval where cumulative exceeds duration
            intervals = intervals[:idx]
            intervals[-1,1] -= cum_duration[idx-1] - duration

    elif duration < 0:
        cum_duration = np.cumsum(np.diff(intervals[::-1])) # backwards cumulative duration
        if cum_duration[-1] > -duration:
            idx = np.searchsorted(cum_duration,-duration) + 1
            intervals = intervals[-idx:]
            intervals[0,0] += cum_duration[idx-1] + duration

    return intervals


def intersectIntervals(intervals):
    """intersect interval sets, obtaining the set of intervals which are contained in all inputs

    arguments:
        intervals       (n) list of (:,2), every element is a set of [start time, stop time] intervals

    output:
        intersection    (:,2) float, intersection between elements of input
    """

    # return single interval unchanged
    if len(intervals) == 1:
        return intervals[0]

    # 1: more than 2 intervals, recursive call
    if len(intervals) > 2:
        intervals = [intervals[0], intersectIntervals(intervals[1:])]

    # 2: intersect 2 interval sets

    # consolidate every interval set
    intervals = [consolidate(i) for i in intervals]

    if intervals[0].size == 0:
        return intervals[0]

    # flatten both sets
    a = intervals[0].flatten()
    b = intervals[1].flatten()

    # ind[i] is odd iff a[i] falls in an interval of b
    ind = np.digitize(a,b)

    # handle case when a ∋ [10,20] and b ∋ [20,35] : interval [20,20] must not be in result
    end_nz = ((ind != 0) & (np.arange(ind.shape[0]) % 2)).astype(bool) # end_nz[i] is 1 iff ind[i] is not 0 and i is odd
    change_ind = a[end_nz] == b[ind[end_nz]-1] # change_ind[j] is 1 iff corresponding interval must be shortened
    find_end_nz = np.where(end_nz)[0]
    ind[find_end_nz[change_ind]] -= 1

    # if ind[2*i-1], ind[2*i] are equal and even, it's an interval of a to exclude entirely
    keep_ind = ((ind[::2] % 2).astype(bool) | (ind[::2] != ind[1::2])).repeat(2)
    ind = ind[keep_ind]
    a = a[keep_ind]
    if a.size == 0:
        return np.zeros((0,2))

    # odd_ind[i] is index of an odd element of ind, which will be replaced by a[odd_ind[i]]
    odd_ind = np.where(ind % 2)[0]
    # round start to lower even number and stop to lower odd number
    start = ind[::2] - ind[::2] % 2
    stop = ind[1::2] -1 + ind[1::2] % 2
    # expand each start[i], stop[i] pair to include intervals in between
    new_ind = np.concatenate([np.arange(start[i],stop[i]+1) for i in range(start.shape[0])]).astype(int)
    # remap odd_ind to point elements of new_ind, which will have to be drawn from a
    remapping = np.array([np.ones(start.shape),stop-start]).flatten('F').cumsum() - 1
    new_odd_ind = remapping[odd_ind].astype(int)

    # initialize intervals as [b[new_ind[i]],b[new_ind[i+1]]]
    intersection = b[new_ind]
    # when ind[j] was odd, replace corresponding element with a[odd_ind[j]]
    intersection[new_odd_ind] = a[odd_ind]
    # return as interval set
    intersection = intersection.reshape((intersection.shape[0]//2,2))

    return intersection


def subtractIntervals(a,b):
    """subtract from an interval set its intersection with a second set

    arguments:
        a, b    (:,2) float, every row is [start time, stop time]

    output:
        c       (:,2) float, a \\ b
    """

    # consolidate and flatten
    a = consolidate(a)
    b = consolidate(b).flatten()
    if a.size == 0 or b.size == 0:
        return a
    a = a.flatten()

    # ind[i] is odd iff a[i] falls in an interval of b
    ind = np.digitize(a,b)

    # if ind[2*i-1], ind[2*i] are equal and odd, it's an interval of a to exclude entirely
    keep_ind = (~(ind[::2] % 2).astype(bool) | (ind[::2] != ind[1::2])).repeat(2)
    ind = ind[keep_ind]
    a = a[keep_ind]

    # even_ind[i] is index of an even element of ind, which will be replaced by a[even_ind[i]]
    even_ind = np.where(ind % 2 == 0)[0]
    # round start to upper even number and stop to upper odd number
    start = ind[::2] + ind[::2] % 2
    stop = ind[1::2] + 1 - ind[1::2] % 2
    # expand each start[i], stop[i] pair to include intervals in between
    new_ind = np.concatenate([np.arange(start[i],stop[i]+1) for i in range(start.shape[0])]).astype(int)
    # remap even_ind to point elements of new_ind, which will have to be drawn from a
    remapping = np.array([np.ones(start.shape),stop-start]).flatten('F').cumsum() - 1
    new_even_ind = remapping[even_ind].astype(int)

    # initialize intervals as [b[new_ind[i]],b[new_ind[i+1]]]
    new_ind -= 1
    new_ind[new_even_ind] = 0 # place holder
    c = b[new_ind]
    # when ind[j] was even, replace corresponding element with a[even_ind[j]]
    c[new_even_ind] = a[even_ind]
    # return as interval set
    c = c.reshape((c.shape[0]//2,2))
    # remove remaining empty intervals
    c = c[(np.diff(c)!=0).ravel(),:]

    return c


def restrict(samples, intervals, shift:bool=None, s_ind:bool=None, i_ind:bool=None):
    """keep only samples falling in a set of intervals

    arguments:
        samples      (:,:,) float, rows are [time stamp, value1, value2, ...]; if 1d, reshaped to (:,1)
        intervals    (:,2) float, rows are [start, stop] times for an interval; if 1d, reshaped to (1,2)
        shift        bool = False, if True, shift remaining epochs together in time
        s_ind        bool = False, if True, return also Is
        i_ind        bool = False, if True, return also Ii

    output:
        samples      (p,m) float, restricted samples, i. e., rows of 'samples' whose first element falls into 'intervals'
        Is           (n) bool, optional, indicese of original samples which were kept
        Ii           (n) bool, optional, indicese of intervals which contain kept samples
    """

    if s_ind is None: s_ind = False
    if i_ind is None: i_ind = False

    # promote 1d arrays to 2d
    samples = np.asarray(samples)
    squeeze = samples.ndim == 1
    if squeeze:
        samples = samples.reshape((-1,1))

    # consolidate intervals to use vectorized algorithm
    intervals = consolidate(intervals)

    # assign and index to each sample time, only odd indeces belong to intervals
    ind = np.digitize(samples[:,0], intervals.flatten())
    Is = (ind % 2).astype(bool) # Is[i] is 1 iff samples[i] must be kept
    samples = samples[Is]
    Ii = (ind[Is] - 1) // 2 # Ii[j] indexes interval that contains samples[Is][j]

    if shift:
        # cumulative shifts: distances between intervals
        ii_distance = intervals[1:,0] - intervals[:-1,1]
        cum_shift = np.concatenate(([0],np.cumsum(ii_distance)))
        # assign cumulative shifts to samples
        shifts = cum_shift[Ii]
        samples[:,0] = samples[:,0] - shifts - intervals[0,0]

    if squeeze:
        samples = samples.reshape(-1)

    if not s_ind and not i_ind:
        return samples
    out = (samples,Is,Ii) # prepare tuple to return requested outputs
    return out[:2 + i_ind:2 - s_ind]


def unshift(samples, intervals):
    """perform the opposite operation of option 'shift' in restrict

    arguments:
        samples      (:,:) float, rows are [time stamp, value1, value2, ...]; if 1d, reshaped to (:,1)
        intervals    (:,2) float, rows are [start, stop] times (s) for an interval, previously used to shift `samples`
    """

    samples = np.asarray(samples, dtype=float)
    intervals = np.array(intervals, ndmin=2)
    # promote 1d arrays to 2d
    if samples.ndim == 1:
        samples = samples.reshape((-1, 1))

    shifted_end = np.cumsum(np.diff(intervals,axis=1).flatten())
    shifted_start = np.concatenate(([0],shifted_end[:-1]))
    shifted_intervals = np.column_stack((shifted_start,shifted_end))

    to_add = intervals[:,0] - shifted_start # offset to add to each sample
    _, is_ok, int_ind = restrict(samples[:,0], shifted_intervals, s_ind=True, i_ind=True)

    samples[~is_ok,0] = np.nan
    samples[is_ok,0] += to_add[int_ind]

    return samples