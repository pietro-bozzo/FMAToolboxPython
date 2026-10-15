""" General-purpose data processing for FMAToolbox """

import numpy as np
import fmatoolbox.intervals


def unwhere(indices,shape=None):
    """perform the inverse operation of ``numpy.where``

    arguments:
        indices    ndarray of int, output of `numpy.where`
        shape      tuple = None, shape of the output array

    output:
        mask       ndarray of bool, original input to `numpy.where`
    """

    if shape is None:
        shape = tuple(i.max() + 1 if len(i) else 0 for i in indices)
    mask = np.zeros(shape,dtype=bool)
    mask[indices] = True
    return mask


def shuffleEvents(events, offset:float=None, n:int=None, group=None, fast:bool=None, intervals=None):
    '''
    shuffle point-process events (i.e., time stamps) preserving their inter-event interval

    arguments:
        events       (s,) | (s,f) float, samples to shuffle; s: # samples, f: # features, if 2d, first column contains time stamps
        offset       float = 0 s, reference time, necessary when events start at a time != 0, e.g., for a portion of a recording
                     in [1000 s, 3000 s]; must be smaller than first event time, ignored if 'intervals' are provided
        n            int = 1, number of independent repetitions of the shuffling procedure, changes output shape
        group        (s,) int, grouping variable to shuffle events separately per group, adds output 'group'
        fast         bool = False, if True, 'events' must be time sorted (to save computation time)
        intervals    (:,2) float = None, rows are [start, stop] times of intervals to restrict shuffle into; if 1d, reshaped to (1,2)

    output:
        shuffled     (:,:,) float, shuffled events, shape depends on 'n':
                     - same as 'events' if 'n' = 1
                     - (s,f,n), i.e., with an extra dimension for 'n' shuffling repetitions, otherwise
        group        (:,) int, optional, grouping variable for shuffled events, only returned if 'group' is not None, shape depends on 'n':
                     - (s,) if 'n' = 1
                     - (s,n), otherwise
    '''

    # default values
    events = np.asarray(events)
    n_dim = events.ndim
    if offset is None: offset = 0
    if n is None: n = 1
    if group is not None: group = np.asarray(group,dtype=int)
    if intervals is not None:
        events, valid = fmatoolbox.intervals.restrict(events,intervals,shift=True,s_ind=True)
        offset = 0
        if group is not None:
            group = group[valid]

    # zero-repetitions case
    if n == 0:
        if group is not None:
            return np.zeros(events.shape+(0,)), np.zeros((events.shape[0],0))
        return np.zeros(events.shape+(0,))

    # make 'events' 3d by appending singleton dimensions
    events = events.reshape(*events.shape, *(1,) * (3 - n_dim)) # (events, features, 1)

    # sort by time
    if not fast:
        order = events[:,0,0].argsort(axis=0)
        events = events[order]
        if group is not None:
            group = group[order]

    # repeat events 'n' times
    events = np.tile(events,(1,1,n)) # (events, features, shuffle repetitions)

    # core f to shuffle events of a group
    rng = np.random.default_rng()
    def shuffleGroup(e):
        # shuffle inter-event intervals (IEI)
        iei = np.diff(e[:,0,:],prepend=offset,axis=0)
        iei_shape = iei.shape
        permutation_idx = np.argsort(rng.random(iei_shape),axis=0)
        iei = iei[permutation_idx,np.arange(iei_shape[1])]
        # reconstruct shuffled samples
        shuffled = offset + np.cumsum(iei,axis=0) # shuffled times (events, shuffle repetitions)
        features = e[permutation_idx[:,None,:], np.arange(1,e.shape[1])[None,:,None], np.arange(iei_shape[1])[None,None,:]]
        shuffled = np.concatenate((shuffled[:,None],features),axis=1) # append shuffled features
        return shuffled

    # perform shuffling
    if group is None:
        shuffled = shuffleGroup(events)
    else:
        shuffled = []
        unique_groups, g_count = np.unique(group,return_counts=True)
        for g in unique_groups:
            shuffled.append(shuffleGroup(events[group==g]))
        # sort by time
        shuffled = np.concatenate(shuffled,axis=0)
        order = shuffled[:,0,:].argsort(axis=0)
        shuffled = shuffled[order[:,None,:], np.arange(shuffled.shape[1])[None,:,None], np.arange(shuffled.shape[2])[None,None,:]]
        # build 'group' for output
        group = np.repeat(unique_groups,g_count) # use unique output to sort
        group = group[order] # sort as 'shuffled'

    # unshift events back into intervals
    if intervals is not None:
        for i in range(n):
            shuffled[:,:,i] = fmatoolbox.intervals.unshift(shuffled[:,:,i],intervals)

    # restore shape
    if n_dim == 1 and n == 1:
        shuffled = shuffled.ravel()
    elif n == 1:
        shuffled = shuffled[:,:,0]

    if group is not None:
        if n_dim == 1 and n == 1:
            group = group.ravel()
        elif n == 1:
            group = group[:,0]
        return shuffled, group

    return shuffled


def circularShift(x, shift, axis:int=None):
    """independent circular shift of each slice of 'x' along a given axis

    arguments:
        x        ndarray to shift
        shift    int | (:,) int, shift amount per slice; if int, the same shift is applied to each slice
        axis     int = 0, axis to roll along; 0 means columns of 'x' will be rolled
    """

    if axis is None: axis = 0
    shift = np.array(shift,ndmin=1)

    n, m = x.shape[:2]
    if axis == 0:
        if shift.size == 1:
            shift = np.repeat(shift,m,axis=0)
        shifted = x[(np.arange(n)[:,None]-shift) % n, np.arange(m)]
    elif axis == 1:
        if shift.size == 1:
            shift = np.repeat(shift,n,axis=0)
        shifted = x[np.arange(n)[:,None], (np.arange(m)-shift[:,None]) % m]
    else:
        raise ValueError('not implemented!')

    return shifted


def jitter(samples,dt,low=None):
    """add a jitter drawn from a uniform distribution to each sample's time stamp

    arguments:
        samples    (s,) | (s,f) numeric, samples to jitter; s: # samples, f: # features, if 2d, first column contains time stamps
        dt         float, higher bound of the uniform distribution
        low        float = -dt, lower bound of the uniform distribution

    output:
        samples    numeric, jittered samples (only time stamps are changed)
    """

    if low is None: low = -dt

    samples = np.asarray(samples,dtype=float)
    n_dim = samples.ndim
    rng = np.random.default_rng()

    if n_dim < 2:
        samples += rng.uniform(low=low,high=dt,size=samples.shape)
    elif n_dim == 2:
        samples[:,0,...] += rng.uniform(low=low,high=dt,size=samples.shape[:1] + samples.shape[2:])

    return samples