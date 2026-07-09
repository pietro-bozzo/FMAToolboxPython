''' Specialized analyses for FMAToolbox '''

import fmatoolbox.general
import numpy as np
import scipy as sp
import sklearn.decomposition as skdc
import skimage.filters as skif
import joblib
import statsmodels.stats.multitest
from typing import Callable


def firingRate(spikes, start:float=None, stop:float=None, bin_size:float=None, step:int=None, smooth:float=None, u_range:tuple[int,int]=None):
    # estimate istantaneous firing rate from spike times
    #
    # arguments:
    #     spikes         (n,:) float, every row is either [spike time] (s) or [spike time, unit id]
    #     start          float = min(spike_times) s, time to start count at
    #     stop           float = max(spike_times) s, time to stop count at
    #     bin_size       float = 0.05 s, time bin to count spikes
    #     step           int = 1, firing rate is computed in windows of length 'bin_size' and overlap 'bin_size' / 'step', default is no overlap
    #     smooth         float = None, gaussian kernel std for smoothing over time
    #     u_range        (2) int, range of units to consider in computation, default is [min(spikes[:,1]), max(spikes[:,1])]
    #                    (boundaries included, only for 2-columns 'spikes')
    #
    # output:
    #     firing_rate    (:,m+1) float, every row is [time stamp, firing rates for m units], m is 1 if spikes has just one column

    # validate input
    spikes = np.asarray(spikes)
    if bin_size is None: bin_size = 0.05
    if step is None: step = 1
    if step % 1 or step == 0:
        raise ValueError("'step' must be a non-zero integer")
    
    units = []
    if spikes.ndim == 1:
        times = spikes
    elif spikes.shape[1] == 1:
        times = spikes.reshape(-1)
    else:
        times = spikes[:,0]
        units = spikes[:,1]
        if u_range is None:
            u_range = np.unique(units).astype(int)[[0,-1]]
        else:
            u_range = np.array(u_range,dtype=int)
    
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
        for u in range(u_range[0],u_range[-1]+1):
            fr = [np.histogram(times[units==u],bins=tb)[0] for tb in time_bins]
            firing_rate.append(np.array(fr).flatten('F'))
        firing_rate = np.array(firing_rate).T / bin_size

    # center times into time bins
    time_bins = [(tb[:-1] + tb[1:]) / 2 for tb in time_bins]
    time_bins = np.array(time_bins).reshape((-1,1),order='F')

    # apply smoothing
    if smooth is not None:
        firing_rate = sp.ndimage.gaussian_filter(firing_rate,smooth,axes=0)

    return np.concatenate((time_bins,firing_rate),1)


def PETH(samples, events, groups=None, g_range:tuple[int,int]=None, limits:tuple[float,float]=None, n_bins:int=None, bin:float=None, step:int=None, fast:bool=False):
    # compute peri-event time histogram of a signal relative to synchronizing events
    #
    # arguments:
    #     samples    float, either:
    #                 - (n) array of time stamps (s), describing a point process
    #                 - (n,:) array, where each row is [time stamp (s), value1, ...], describing one or more continous signals
    #     events     (m) float, synchronizing events' times, their order is maintained in the output 'mat'
    #     groups     (n) int, grouping indeces for samples, to compute separate PETHs (only for point process 'samples')
    #     g_range    (2) int = [0,max(groups)], min and max group id
    #     groups     (:) int, grouping indeces for samples, to compute separate PETHs (only for point process 'samples')
    #     limits     (2) float = [-0.5,0.5] (s), defines a window around events, divided into 'n_bins' time bins to compute PETH
    #     n_bins     float = 101, number of time bins around event times
    #     bin        float = None (s), bin size, can be given instead of 'n_bins', which will be deduced from 'bin' and 'limits'
    #     step       int = 1, only for point-process 'samples', for values higher than 1, time bins inside a window will overlap, yielding:
    #                 - bin_size of (limit[1]-limit[0]) / n_bins, unchanged
    #                 - time resolution of bin_size / step
    #     fast       bool = False, if True, 'samples' must be time sorted to save computation time (only for point process 'samples')
    #
    # output:
    #     mat        (m,n_bins) float, every row corresponds to samples centered on an event
    #     t          (n_bins) float, times (s)
    #     m          (n_bins) float, average samples across events

    # default values
    samples = np.array(samples,ndmin=1)
    point_process = samples.ndim == 1 or samples.shape[1] == 1
    events = np.array(events,ndmin=1)
    if point_process:
        if groups is None:
            groups = np.zeros(samples.shape[0],dtype=int)
            n_groups = 1
            g_range = None
            squeeze = True  # squeeze 'mat' to 2d when 'groups' is None
        else:
            groups = np.array(groups,ndmin=1,dtype=int)
            if samples.shape[0] != groups.shape[0]:
                raise ValueError("'samples' and 'groups' must have the same length")
            n_groups = groups.max() + 1
            squeeze = False
        if g_range is not None:
            groups = groups[(groups >= g_range[0]) & (groups <= g_range[1])]
            groups -= int(g_range[0])
            n_groups -= int(g_range[0])
    else:
        squeeze = samples.shape[1] < 3 # squeeze 'mat' to 2d when 'samples' has only one signal column
    if limits is None: limits = [-0.5,0.5]
    if n_bins is None: n_bins = 101 if bin is None else round((limits[1] - limits[0]) / bin)
    if step is None: step = 1
    
    # sort by time
    if not fast:
        sort_idx = np.argsort(samples) if samples.ndim == 1 else np.argsort(samples[:,0])
        samples = samples[sort_idx]
        if point_process:
            groups = groups[sort_idx]
    
    # 1: point process
    if point_process:

        # build time bins
        t = np.linspace(limits[0],limits[1],n_bins+1)
        t = (t[:-1] + t[1:]) / 2
        t = np.linspace(t[0],t[-1],(n_bins-1)*step+1)
        bin_width = (limits[1] - limits[0]) / n_bins
        mat = np.zeros((len(events), n_bins*step, n_groups), dtype=int)

        for i in range(step):
            stride = i * bin_width / step

            # find where events fall in samples
            left = np.searchsorted(samples, events+limits[0]+stride, side='left')
            right = np.searchsorted(samples, events+limits[1]+stride, side='right')
            counts = right - left
            valid = counts > 0
            # repeat event indices according to how many samples they match
            event_idx = np.repeat(np.arange(len(events))[valid],counts[valid])
            sample_idx = np.concatenate([np.arange(l,r) for l, r in zip(left[valid],right[valid])])
            # build lists of matches
            e_rep = events[event_idx]
            s_sel = samples[sample_idx]
            g_sel = groups[sample_idx]
            # assign matches to 'mat'
            bin_ind = ((s_sel-e_rep-limits[0]-stride) / bin_width).astype(int)
            bin_ind = np.clip(bin_ind, 0, n_bins - 1) # avoid numerical error
            np.add.at(mat, (event_idx, bin_ind*step+i, g_sel), 1)

        # if step != 1, discard last bins which go outisde limits
        mat = mat[:,:(n_bins-1)*step+1,:]

    # 2: time series
    else:
        # build time bins
        t = np.linspace(limits[0],limits[1],n_bins)
        # interpolate PETH matrix
        t_mat = events.reshape((-1,1)) + t.reshape((1,-1)) # interpolation times around events
        mat = np.stack( [np.interp(t_mat,samples[:,0],samples[:,i]) for i in range(1,samples.shape[1])], axis=-1)

    # restore correct 'mat' shape
    if squeeze:
        mat = mat.reshape(mat.shape[:2])

    m = np.mean(mat,axis=0)

    return mat, t, m


def avalanchesFromProfile(x, threshold:float, time_step:float, t0:float=0):
    # compute avalanches' sizes and [start,stop] intervals from a time series
    #
    # arguments:
    #     x            (:) float, time series uniformly sampled in time
    #     threshold    float in [0,100] (%), percentile of x used as a threshold
    #     time_step    float, time distance (s) between two consecutive elements of x
    #     t0           float = 0 (s), time corresponding to first element of x
    #
    # output:
    #     sizes        (n) float, avalanche sizes
    #     intervals    (n,2) float, each row is an avalanche's [start, stop] interval (s)
    #     size_t       (m) float, size over time, in which every avalanche is separated by a 0

    x = np.array(x)

    # threshold the signal
    threshold = np.percentile(x, threshold)
    x = x - threshold
    x[x < 0] = 0

    is_ok = np.concatenate(([True], (x[1:] != 0) | (x[:-1] != 0)))  # is_ok[i] = 0 if i-th element is repeated zero

    # sizes
    size_t = x[is_ok] * time_step  # remove repeated zeros, obtaining size per bin: size over time
    sizes = np.bincount(np.cumsum(size_t == 0) - (x[0] == 0), weights=size_t)
    # remove last zero
    if sizes[-1] == 0:
        sizes = sizes[:-1]

    #  start and stop times
    start = np.where(np.concatenate(([x[0] != 0], (x[1:] != 0) & (x[:-1] == 0))))[0]
    stop = np.where(np.concatenate(((x[1:] == 0) & (x[:-1] != 0), [x[-1] != 0])))[0] + 1
    intervals = np.stack((start, stop), 1) * time_step + t0 - time_step / 2

    return sizes, intervals, size_t


def cellAssembliesICA(spikes, window:float=None, when=None, drop_mix:bool=False):
    # detect assemblies from spike trains with PCA + ICA

    if window is None: window = 0.025

    raster = firingRate(spikes,bin_size=window)
    raster[:,1:] *= window # convert to counts

    time = raster[:,0]
    n = raster[:,1:] # discard time column
    if when is not None:
        time, valid = fmatoolbox.general.restrict(time,when,s_ind=True)
        n = n[valid]

    # remove units which never spiked to avoid cov error
    keep = ~(n==0).all(axis=0)
    n = n[:,keep]

    # correlation matrix
    n = sp.stats.zscore(n,axis=0)
    n_times, n_units = n.shape
    corr = np.cov(n.T)
    eigenvalues, eigenvectors = sp.linalg.eigh(corr) # each column of 'eigenvectors' is an eigenvector

    # keep only significant eigenvectors according to MP distribution criteria CITE PAPER
    q = n_times / n_units
    lambda_max = (1 + np.sqrt(1 / q)) ** 2
    #lambda_max += n_units**(-2/3) # Tracy-Widom correction
    significant = eigenvalues > lambda_max
    eigenvalues = eigenvalues[significant]
    eigenvectors = eigenvectors[:,significant]

    # run ICA
    projection = ((eigenvectors @ eigenvectors.T) @ n.T).T
    n_components = sum(significant)
    ica = skdc.FastICA(n_components=n_components,max_iter=1000)
    ica = ica.fit(projection)
    weights = ica.components_.T # (units, components)

    # normalize weights as in Van de Ven et al (2016)
    weights /= np.linalg.norm(weights,axis=0)

    # sort by variance of the projected signals, which is NOT explained variance per component (as they are not orthogonal)
    activity = n @ weights # (times, components)
    variance = activity.var(axis=0,ddof=1) / n_units
    order = np.argsort(-variance)
    variance = variance[order]
    weights = weights[:,order]

    # identify assembly members (will have to choose one of two methods...)
    weights_otsu = weights.copy()
    weights_morici = weights.copy()
    # 1. Otsu threhsolding
    for c in range(n_components):
        w = weights[:,c]
        thresh = skif.threshold_otsu(np.abs(w))
        mask = np.abs(w) > thresh
        weights_otsu[~mask,c] = 0
    # 2. thresholding from Morici et al (2026), identifying features with an above average contrubtion (if weigth vectors have unit norm, all elements are 1 / np.sqrt(n_units) for a "uniform" vector)
    mask = np.abs(weights) > 1 / np.sqrt(n_units)
    weights_morici[~mask] = 0
    weights = weights_otsu

    # keep only components with no negative "strong" weights
    if drop_mix:
        remove = np.any(weights < 0,axis=0)
        weights = weights[:,~remove]
        eigenvalues = eigenvalues[~remove]
        n_components = sum(~remove)

    # flip signs (as signs are defined up to a per-component flip)
    #flip = weights.max(axis=0) < -weights.min(axis=0) # let argmax(abs( )) be positive
    flip = np.sum(weights > 1e-7,axis=0) < np.sum(weights < -1e-7,axis=0)  # let most elements be positive
    weights[:,flip] *= -1

    # reintroduce units which never spiked
    if not keep.all():
        weights_old = weights.copy()
        weights = np.zeros((len(keep),n_components))
        weights[keep,:] = weights_old
        n_units = len(keep)

    # templates, note that they are independent to the sign flip of weight vectors
    templates = np.empty((n_units,n_units,n_components))
    for i in range(n_components):
        template = np.outer(weights[:,i],weights[:,i])
        np.fill_diagonal(template,0)  # remove the diagonal
        templates[:,:,i] = template

    return weights, templates, raster


def reactivationStrength(raster, templates, threshold:float=5):
    # compute reactivation strength of assemblies as quadratic forms between raster and templates

    def template_strength(template):
        return np.nansum(raster * (raster @ template), axis=1)

    time = raster[:,0]
    raster = raster[:,1:]

    # following Morici et al. (2026), smooth and z-score raster
    raster = sp.ndimage.gaussian_filter(raster,0.5,axes=0)
    raster = sp.stats.zscore(raster,axis=0)

    n_templates = templates.shape[2]
    strength = np.column_stack(joblib.Parallel(n_jobs=-1)(joblib.delayed(template_strength)(templates[:,:,i]) for i in range(n_templates)))

    # following Morici et al. (2026), peaks are avalanches in reactivation with threshold 5
    peaks = []
    for col in range(strength.shape[1]):
        indices, properties = sp.signal.find_peaks(strength[:,col],height=threshold)
        peaks.append(np.column_stack((time[indices],strength[indices,col])))

    strength = np.column_stack((time,strength))

    return strength, peaks


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


def holmBonferroni(pvals, alpha:float=0.05, return_reject:bool=False):
    '''
    Holm-Bonferroni correction for multiple tests

    arguments:
        pvals            (n,) float, p values, NaNs are ignored in the correction procedure and propagated in output
        alpha            float = 0.05, significance level, must be in [0,1]
        return_reject    bool = False, return also rejection decisions

    output:
        corrected        (n,) float, adjusted p values
        reject           (n,) bool, true for hypothesis that can be rejected
    '''

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


def maxStatisticTest(data, surrogate, statistic=None, group=None, alpha:float=0.05, alternative:str='two-sided'):
    '''
    conduct a max statistic test over time, assessing in which time points the null hypothesis about a statistic across sessions can be rejected

    arguments:
        data           (sessions, times) float
        surrogate      (sessions, times, surrogates) float
        group          (sessions,) int, grouping variable used to aggregate sessions, the statistic is computed per group and then again over groups
        alpha          float = 0.05, significance level, must be in [0,1]
        alternative    str = {'two-sided','grater','less'}, test direction, defines the null hypothesis
    '''

    data = np.array(data,ndmin=2)
    surrogate = np.array(surrogate,ndmin=3)
    if data.shape[:2] != surrogate.shape[:2]:
        raise ValueError("'data' and 'surrogate' must have the same first two dimensions (sessions, times)")
    if statistic is None:
        statistic = lambda x : np.nanmean(x,axis=0)
    n_times = data.shape[1]
    n_surrogates = surrogate.shape[2]

    # statistic for real and surrogate data
    if group is None:
        s_real = statistic(data) # (times,)
        s_surrogate = statistic(surrogate) # (times, surrogates)
    else:
        unique_groups = np.unique(group)
        s_real = [statistic(data[group==g]) for g in unique_groups]
        s_real = statistic(s_real)
        s_surrogate = [statistic(surrogate[group==g]) for g in unique_groups]
        s_surrogate = statistic(s_surrogate)

    # p-values per time point
    if alternative == 'greater':
        s_surrogate = np.min(s_surrogate,axis=0) # (surrogates,)
        p = MCpValue(np.tile(s_surrogate,(n_times,1)).T,s_real,alternative) # (times,)
    elif alternative == 'less':
        s_surrogate = np.max(s_surrogate,axis=0)
        p = MCpValue(np.tile(s_surrogate,(n_times,1)).T,s_real,alternative)
    elif alternative == 'two-sided':
        # standardize statistic to ensure proper two-tailed test
        mu = np.mean(s_surrogate,axis=1) # (times,)
        sigma = np.std(s_surrogate,axis=1,ddof=1)
        s_real = np.abs((s_real - mu) / sigma) # abs(z-score( ))
        s_surrogate = (s_surrogate - mu.reshape(-1,1)) / sigma.reshape(-1,1)
        s_surrogate = np.max(np.abs(s_surrogate),axis=0) # max_t(abs(z-score( ))), i.e., (surrogates,)
        p = MCpValue(np.tile(s_surrogate,(n_times,1)).T,s_real,'greater')
    else:
        raise ValueError("'alternative' must be 'two-sided', 'greater' or 'less'")

    return p < alpha