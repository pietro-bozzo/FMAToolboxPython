''' Specialized analyses for FMAToolbox '''

import numpy as np
from numpy.ma.core import squeeze
from scipy.ndimage import gaussian_filter
import statsmodels.stats.multitest
from typing import Callable


def firingRate(spikes,start=None,stop=None,bin_size=0.05,step=1,smooth=None):
    # estimate istantaneous firing rate from spike times
    #
    # arguments:
    #     spikes         (n,:) float, every row is either [spike time] (s) or [spike time, unit id]
    #     start          float = min(spike_times) s, time to start count at
    #     stop           float = max(spike_times) s, time to stop count at
    #     bin_size       float = 0.05 s, time bin to count spikes
    #     step           int = 1, firing rate is computed in windows of length 'binSize' and overlap 'binSize' / 'step',
    #                    default is no overlap
    #
    # output:
    #     firing_rate    (:,m+1) float, every row is [time stamp, firing rates for m units], m is 1 if spikes has just one column

    # validate input
    try:
        spikes = np.array(spikes)
    except Exception as e:
        raise e
    if step % 1 or step == 0:
        raise ValueError('\'step\' must be a non-zero integer')
    
    units = []
    if spikes.ndim == 1:
        times = spikes
    elif spikes.shape[1] == 1:
        times = spikes.reshape(-1)
    else:
        times = spikes[:,0]
        units = spikes[:,1]
    
    # build time bins, overlapping if requested
    if start is None:
        start = times.min()
    if stop is None:
        stop = times.max()
    time_bins = [np.arange(start,stop+bin_size,bin_size) + i*bin_size/step for i in range(step)]
    
    if len(units) == 0:
        # compute firing rate once
        firing_rate = [np.histogram(times,bins=tb)[0] for tb in time_bins]
        # flatten and convert to Hz
        firing_rate = np.array(firing_rate).reshape((-1,1),order='F') / bin_size
    else:
        # compute firing rate once per unit and stack into a matrix
        firing_rate = []
        for u in np.unique(units):
            fr = [np.histogram(times[units==u],bins=tb)[0] for tb in time_bins]
            firing_rate.append(np.array(fr).flatten('F'))
        firing_rate = np.array(firing_rate).T / bin_size

    # center times into time bins
    time_bins = [(tb[:-1] + tb[1:]) / 2 for tb in time_bins]
    time_bins = np.array(time_bins).reshape((-1,1),order='F')

    # apply smoothing
    if smooth is not None:
        firing_rate = gaussian_filter(firing_rate,smooth,axes=0)

    return np.concatenate((time_bins,firing_rate),1)


def PETH(samples,events,groups=None,g_range=None,limits=[-0.5,0.5],n_bins=101,fast=False):
    # compute peri-event time histogram of a signal relative to synchronizing events
    #
    # arguments:
    #     samples    (:,:) float, every row is either [time stamps] (s) or [time stamps, value]
    #     events     (:) float, synchronizing events' times, CHECK WHAT HAPPENS FOR NON SORTED WITH 2 COLS SAMPLES
    #     groups     (:) int, grouping indeces for samples, to compute separate PETHs (only for vector samples)
    #     g_range    (2) int = [0,max(groups)], min and max group id
    #     limit      (2) float = [-0.5,0.5] (s), defines window around events to compute PETH
    #     n_bins     float = 101, number of time bins around event times
    #     fast       bool = False, if True, samples are expected to be time sorted (to save time)
    #
    # output:
    #     mat        (n,n_bins) float, every row is samples centered on an event
    #     t          (1,n_bins) float, times (s)
    #     m          (n,1) float, average samples across events

    samples = np.array(samples,ndmin=1)
    events = np.asarray(events)
    squeeze = True
    if groups is None:
        groups = np.zeros(samples.shape[0],dtype=int)
        n_groups = 1
        g_range = None
    else:
        groups = np.array(groups,ndmin=1,dtype=int)
        if samples.shape[0] != groups.shape[0]:
            raise ValueError("samples and groups must have the same length")
        n_groups = groups.max() + 1
        squeeze = False
    if g_range is not None:
        groups = groups[(groups >= g_range[0]) & (groups <= g_range[1])]
        groups -= int(g_range[0])
        n_groups -= int(g_range[0])
    
    # sort by time
    if not fast:
        samples = np.sort(samples) if samples.ndim == 1 else samples[samples[:,0].argsort()]
    
    # 1: point process
    if samples.ndim == 1 or samples.shape[1] == 1:

        # build time bins
        t = np.linspace(limits[0],limits[1],n_bins+1)
        t = (t[:-1] + t[1:]) / 2
        bin_width = np.diff(limits) / n_bins

        mat = np.zeros((len(events), n_bins, n_groups), dtype=int)
        # find where events fall in samples
        left = np.searchsorted(samples, events + limits[0], side='left')
        right = np.searchsorted(samples, events + limits[1], side='right')
        counts = right - left
        valid = counts > 0
        # repeat event indices according to how many samples they match
        event_idx = np.repeat(np.arange(len(events))[valid],counts[valid])
        sample_idx = np.concatenate([np.arange(l,r) for l, r in zip(left[valid],right[valid])])

        e_rep = events[event_idx]
        s_sel = samples[sample_idx]
        g_sel = groups[sample_idx]

        bin_ind = ((s_sel - e_rep - limits[0]) / bin_width).astype(int)
        bin_ind = np.clip(bin_ind, 0, n_bins - 1)

        np.add.at(mat, (event_idx, bin_ind, g_sel), 1)

        # find where events fall in samples OLD
        # left = np.searchsorted(samples,events+limits[0],side='left')
        # right = np.searchsorted(samples,events+limits[1],side='right')
        # valid = right - left > 0
        # mat = np.zeros((len(events),n_bins,n_groups),dtype=int)
        # for i, e in enumerate(events):
        #     if valid[i]:
        #         distance = samples[left[i]:right[i]] - e
        #         bin_ind = ((distance-limits[0])/bin_width).astype(int) # USE clip
        #         bin_ind[bin_ind < 0] = 0
        #         bin_ind[bin_ind >= n_bins] = n_bins - 1
        #         np.add.at(mat,(i,bin_ind,groups[left[i]:right[i]]),1)

        if squeeze:
            mat = mat.reshape((mat.shape[:2]))

    # 2: time series
    else:
        # build time bins
        t = np.linspace(limits[0],limits[1],n_bins)
        # interpolate PETH matrix
        t_mat = events.reshape((-1,1)) + t.reshape((1,-1)) # interpolation times around events
        mat = np.interp(t_mat,samples[:,0],samples[:,1])

    m = np.mean(mat,axis=0)

    return mat, t, m


# --- statistics functions ---

def MCpValue(surrogate,real,alternative='two-sided'):
    """
    Compute Monte Carlo p-values comparing real statistics to surrogate distributions

    Parameters
    ----------
    surrogate : array_like, shape (n_surrogates, n_features)
        surrogate statistics
    real : array_like, shape (n_features,)
        observed statistics
    alternative : {"two-sided", "greater", "less"}
        direction of the test

    Returns
    -------
    pvals : ndarray, shape (n_features,)
        Monte Carlo p-values
    """

    surrogate = np.asarray(surrogate)
    real = np.asarray(real).ravel()
    if surrogate.ndim == 1:
        surrogate = surrogate.reshape((-1,1))
    if surrogate.shape[1] != real.shape[0]:
        raise ValueError("real must have one element for every column of surrogates")
    
    if alternative == "greater":
        count = np.sum(surrogate >= real, axis=0)

    elif alternative == "less":
        count = np.sum(surrogate <= real, axis=0)

    elif alternative == "two-sided":
        greater = np.sum(surrogate >= real, axis=0)
        less = np.sum(surrogate <= real, axis=0)
        count = 2 * np.minimum(greater, less)

    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    
    pvals = (count + 1) / (surrogate.shape[0] + 1) # +1 implement finite-sample Monte Carlo correction
    
    return np.minimum(pvals, 1.0)


def holmBonferroni(pvals,alpha=0.05,return_reject=False):
    """
    Holm-Bonferroni correction for multiple tests

    Parameters
    ----------
    pvals : array-like 
        array of p-values, NaNs are ignored in the correction procedure and propagated in output
    alpha : float = 0.05
        significance level
    return_reject : bool = False
        whether to also return rejection decisions

    Returns
    -------
    corrected : ndarray
        adjusted p-values, preserves input shape
    reject : ndarray of bool, optional
        true for hypothesis that can be rejected for given alpha, preserves input shape
    """

    pvals = np.asarray(pvals)
    original_shape = pvals.shape
    flat = pvals.ravel()
    valid_mask = np.isfinite(flat) # valid (non-NaN) p-values
    corrected_flat = np.full_like(flat,np.nan,dtype=float)
    reject_flat = np.full_like(flat,False,dtype=bool)

    if valid_mask.any():
        reject, corrected, _, _ = statsmodels.stats.multitest.multipletests(flat[valid_mask],alpha=alpha,method="holm")
        corrected_flat[valid_mask] = corrected
        reject_flat[valid_mask] = reject

    # restore original shape
    corrected = corrected_flat.reshape(original_shape)
    reject = reject_flat.reshape(original_shape)

    if return_reject:
        return corrected, reject
    return corrected


def maxStatisticTest(data, surrogate, statistic=None, alpha:float=0.05, alternative:str='two-sided'):
    # conduct a max statistic test over time
    #
    # arguments:
    #     data (n sessions, n times)
    #     surrogate (n session, n times, n surrogates)
    
    data = np.array(data,ndmin=2)
    surrogate = np.array(surrogate,ndmin=2)
    if data.shape[:2] != surrogate.shape[:2]:
        raise ValueError("'data' and 'surrogate' must have the same first two dimensions")
    if surrogate.ndim != 3:
        raise ValueError("'surrogate' must have dimensions (sessions, times, surrogates)")
    if statistic is None:
        statistic = lambda x : np.nanmean(x,axis=0)
    n_times = data.shape[1]
    n_surrogates = surrogate.shape[2]

    # statistic for real and surrogate data
    s_real = statistic(data) # (n_times,)
    s_surrogate = np.zeros((n_times,n_surrogates)) # (n_times,n_surrogates)
    for i in range(n_surrogates):
        s_surrogate[:,i] = statistic(surrogate[:,:,i])

    # p-values per time point
    if alternative == 'greater':
        s_surrogate = np.min(s_surrogate,axis=0) # (n_surrogates,)
        p = MCpValue(np.tile(s_surrogate,(n_times,1)).T,s_real,alternative) # (n_times,)
    elif alternative == 'less':
        s_surrogate = np.max(s_surrogate,axis=0)
        p = MCpValue(np.tile(s_surrogate,(n_times,1)).T,s_real,alternative)
    elif alternative == 'two-sided':
        # standardize statistic to ensure proper two-tailed test
        mu = np.mean(s_surrogate,axis=1) # (n_times,)
        sigma = np.std(s_surrogate,axis=1,ddof=1)
        s_real = np.abs((s_real - mu) / sigma) # abs(z-score( ))
        s_surrogate = (s_surrogate - mu.reshape(-1,1)) / sigma.reshape(-1,1)
        s_surrogate = np.max(np.abs(s_surrogate),axis=0) # max_t(abs(z-score( ))), i.e., (n_surrogates,)
        p = MCpValue(np.tile(s_surrogate,(n_times,1)).T,s_real,"greater")
    else:
        raise ValueError("'alternative' must be 'two-sided', 'greater' or 'less'")

    return p < alpha