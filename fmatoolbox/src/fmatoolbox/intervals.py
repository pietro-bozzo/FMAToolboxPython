""" Utilities to handle time intervals for FMAToolbox """

import numpy as np


def consolidate(intervals,eps:float=None, duration:float=None):
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