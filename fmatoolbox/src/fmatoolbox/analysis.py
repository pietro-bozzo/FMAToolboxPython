''' Specialized analyses for FMAToolbox '''

import fmatoolbox.general
import numpy as np
import scipy as sp
import statsmodels.stats.multitest
from numba import njit
from typing import Literal, Callable


def istantaneousRate(samples, start:float=None, stop:float=None, bin:float=None, step:int=None, smooth:float=None, g_range:tuple[int,int]=None):
    """ estimate the istantaneous rate of a point process from a realization of its time stamps, e.g., the firing rate from spike times

    arguments:
        samples        (n,) float | (n,2) float, every row is either [sample time (s)] or [sample time (s), process id]; if 1d, interpreted as (n,1)
        start, stop    float = min(samples[:,0]), max(samples[:,0]) s, time to start / stop count at (s)
        bin            float = 0.05 s, time bin to count samples
        step           int = 1, rate is computed in windows of length 'bin' and overlap 'bin' / 'step', default is no overlap
        smooth         float = None, gaussian kernel std to smooth rate over time
        g_range        (2,) int = [min(samples[:,1]), max(samples[:,1])], range of process ids to consider (boundaries included,
                       only for 2-columns 'samples')

    output:
        rate           (:,g+1) float, every row is [time stamp (s), rates for g processes], g is 1 if 'samples' has just one column
    """

    # validate input
    samples = np.asarray(samples)
    if bin is None: bin = 0.05
    if step is None: step = 1
    if step % 1 or step == 0:
        raise ValueError("'step' must be a non-zero integer")

    # parameters
    if samples.ndim == 1 or samples.shape[1] == 1:
        times = samples.ravel()
        groups = np.zeros_like(times,dtype=np.int64)
        g_min = 0
        g_max = 0
    else:
        times = samples[:,0]
        groups = samples[:,1].astype(np.int64)
        if g_range is None:
            g_min, g_max = np.unique(groups).astype(int)[[0,-1]]
        else:
            g_min = int(g_range[0])
            g_max = int(g_range[-1])
    if start is None: start = times.min()
    if stop is None: stop = times.max()
    n_bins = int((stop + bin - start) // bin)

    rate = _hist_numba(times, float(start), float(bin), int(step), n_bins, groups, g_min, g_max)
    rate = rate.reshape(n_bins*step,-1) / bin # convert to rate (Hz), flatten along steps
    t = start + bin/2 + np.linspace(0,n_bins*bin - bin/step,n_bins*step) # make time axis

    # apply smoothing
    if smooth is not None:
        rate = sp.ndimage.gaussian_filter(rate,smooth,axes=0)

    return np.column_stack((t,rate))


@njit
def _hist_numba(times, start, bin, step, nbins, groups, g_min, g_max):
    """ build histograms corresponding to multiple temporally shifted windows, counting grouped samples

    arguments:
        times           (n,) float, timestamps
        nbins           int, number of time bins
        bin             float, bin width (s)
        start           float, start time of the analysis (s)
        step            int, number of temporal offsets (overlapping windows)
        groups          (n,) int, group id associated with each timestamp
        g_min, g_max    int, min / max group id included

    output:
        hist            (nbins,step,n_groups) int, histogram counts for each bin, temporal offset, and group
    """

    nspikes = len(times)
    ngroups = g_max - g_min + 1
    gap = bin / step

    hist = np.zeros((nbins,step,ngroups),dtype=np.int64)
    for i in range(nspikes):
        gi = groups[i]
        if gi >= g_min and gi <= g_max:
            gi -= g_min
            ti = times[i] - start # shift to begin counting at 'start'
            for j in range(step):
                bin_idx = int((ti - j * gap) / bin)
                if bin_idx >= 0 and bin_idx < nbins:
                    hist[bin_idx,j,gi] += 1

    return hist


def histogramVectorized(x,nbins):
    # NOTE: appears to be faster than _hist_numba, but cannot handle groups
    a, b = x.shape
    # assign bin indeces
    bin_idx = (x / bin).astype(int) # bin_idx = 0, bin_idx = n_bins + 1 after 'stop'
    bin_idx = np.clip(bin_idx,0,nbins-1)
    offset = np.arange(b)[None,:] * nbins
    hist = np.bincount((bin_idx+offset).ravel(), minlength=nbins*b).reshape(b,nbins).T
    return hist


def PETH(samples, events, groups=None, g_range:tuple[int,int]=None, limits:tuple[float,float]=None, n_bins:int=None, bin:float=None, step:int=None, smooth:float=None, fast:bool=False):
    '''
    compute peri-event time histogram of a signal relative to synchronizing events

    arguments:
        samples    float, either:
                    - (n,) array of time stamps (s), describing a point process
                    - (n,:) array, where each row is [time stamp (s), value1, ...], describing one or more continous signals
        events     (m,) float, synchronizing events' times, their order is maintained in the output 'mat'
        groups     (n,) int, grouping indeces for samples, to compute separate PETHs (only for point process 'samples')
        g_range    (2,) int = [0,max(groups)], min and max group id
        groups     (:,) int, grouping indeces for samples, to compute separate PETHs (only for point process 'samples')
        limits     (2,) float = [-0.5,0.5] (s), defines a window around events, divided into 'n_bins' time bins to compute PETH
        n_bins     float = 101, number of time bins around event times
        bin        float = None (s), bin size, can be given instead of 'n_bins', which will be deduced from 'bin' and 'limits'
        step       int = 1, only for point-process 'samples', for values higher than 1, time bins inside a window will overlap, yielding:
                    - bin_size of (limit[1]-limit[0]) / n_bins, unchanged
                    - time resolution of bin_size / step
        smooth     float = None, gaussian kernel std to smooth mean output 'm' over time, has no effect on 'mat'
        fast       bool = False, if True, 'samples' must be time sorted to save computation time (only for point process 'samples')

    output:
        mat        (m,n_bins) float, every row corresponds to samples centered on an event
        t          (n_bins,) float, times (s)
        m          (n_bins,) float, average of 'samples' across events
    '''

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
            g_range = np.sort(g_range,axis=None)
            groups = groups[(groups >= g_range[0]) & (groups <= g_range[-1])]
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
            sample_idx = np.concatenate([np.arange(l,r) for l, r in zip(left[valid],right[valid])]) if np.any(valid) else []
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

    m = np.nanmean(mat,axis=0)
    if smooth is not None:
        m = sp.ndimage.gaussian_filter(m,smooth,axes=0)

    return mat, t, m


def jointPETH(samples, events, bin:float=None, step:int=None, n_bins=None, limits=None, return_peths:bool=None, smooth:float=None):
    '''
    compute joint peri-event time histogram of two signals A and B relative to synchronizing events, i.e., the co-occurrence rate (Hz)
    for different time-lag pairs applied to A and B

    arguments:
        samples       (2,) tuple, whose elements can be either:
                      - (n) float of time stamps (s), describing a point process
                      - (n,:) float, where each row is [time stamp (s), value1, ...], describing one or more continous signals
        events        (m) float, synchronizing events' times, their order is maintained in the output 'mat' /!\
        bin           float = 0.05 (s), bin size to compute PETHs
        n_bins        int | (2,) int = 101, number of time bins around event times, either one for both PETHS or two values
        limits        (2,) float | (2,2) float = None, can be given instead of 'n_bins', which will be deduced from 'bin' and 'limits', and
                      defines a [start, stop] window (in s) around events, divided into time bins of size 'bin' to compute PETHs; can be a
                      single tuple for both PETHS or a tuple of tuples
        step          int = 1, only for point-process 'samples', for values higher than 1, time bins inside each window will overlap, yielding:
                       - bin_size of (limit[1]-limit[0]) / n_bins, unchanged
                       - time resolution of bin_size / step
        smooth        float = None, gaussian kernel std to smooth mean output 'peth' over time, has no effect on other outputs
        ARGS YET TO TEST
        groups        (n) int, grouping indeces for samples, to compute separate PETHs (only for point process 'samples')
        g_range       (2) int = [0,max(groups)], min and max group id
        groups        (:) int, grouping indeces for samples, to compute separate PETHs (only for point process 'samples')
        fast          bool = False, if True, 'samples' must be time sorted to save computation time (only for point process 'samples')

    output:
        joint         (n_bins0,n_bins1) float, lagged co-occurrency rate (Hz), dimensions correspond to time lags w.r.t. 'events' for
                      A and B, respectively
        null          (n_bins0,n_bins1) float, null model for 'joint' in case A and B occurr independently, conditioned on 'events';
                      i.e., their conditioned co-occurrence probability is the product of their average event PETHs
        difference    (n_bins0,n_bins1) float, 'joint' - 'null', co-occurrence rate (Hz) unexplained by the null model, values far from 0 suggest
                      that conditioning on 'events' is not sufficient to explain the co-occurrence of the two signals
    '''

    # 1. ensure PETHs are produced with the same bin size
    isscalar = lambda x: x is None or np.isscalar(x)
    if bin is None: bin = 0.05
    # duplicate single-peth inputs for two peths
    if isscalar(n_bins):
        n_bins = [n_bins,n_bins]
    else:
        n_bins = list(n_bins)
    if limits is None:
        limits = [None,None]
    else:
        limits = [list(limits),list(limits)] if np.isscalar(limits[0]) and np.isscalar(limits[1]) else [l if isscalar(l) else list(l) for l in limits]
    # deduce 'n_bins' or 'limits'
    for i in [0,1]:
        if n_bins[i] is None:
            if limits[i] is None:
                n_bins[i] = 101
                limits[i] = (-bin * n_bins[i] / 2, bin * n_bins[i] / 2)
            else:
                window = np.diff(limits[i]).item()
                n_bins[i] = np.ceil(window / bin).astype(int)
                extra = n_bins[i] * bin - window
                limits[i][0] -= extra / 2
                limits[i][1] += extra / 2
        else:
            limits[i] = (-bin * n_bins[i] / 2, bin * n_bins[i] / 2)

    # 2. PETHs
    mat0, t0, mean0 = PETH(samples[0],events,limits=limits[0],n_bins=n_bins[0],bin=bin,step=step)
    mat1, t1, mean1 = PETH(samples[1],events,limits=limits[1],n_bins=n_bins[1],bin=bin,step=step)
    dt = t0[1] - t0[0]
    if not np.isclose(dt, t1[1]-t1[0]):
        raise ValueError('time resolution of the two PETHs must coincide')

    # observed co-occurrence between 'samples0' and 'samples1' at every time bin
    joint = (mat0.T @ mat1) / len(events)
    # expected co-occurrence if the signals were independent: product
    null = np.outer(mean0, mean1)
    # convert to Hz
    joint = np.sqrt(joint) / dt
    null = np.sqrt(null) / dt
    # difference(i,j) = sqrt[ mean_e( mat0(e,i) * mat1(e,j) ) ] - sqrt[ mean0(i) * mean1(j) ]
    difference = joint - null

    if return_peths:
        if smooth is not None:
            # recompute with smoothing
            _, _, mean0 = PETH(samples[0],events,limits=limits[0],n_bins=n_bins[0],bin=bin,step=step,smooth=smooth)
            _, _, mean1 = PETH(samples[1],events,limits=limits[1],n_bins=n_bins[1],bin=bin,step=step,smooth=smooth)
        mean0 /= dt
        mean1 /= dt
        return joint, null, difference, t0, t1, mean0, mean1
    return joint, null, difference, t0, t1


@njit
def _ccg_numba(times, proc, nproc, bin_width, lag_start, lag_stop):
    # ccg: (reference process, target process, lag bin)

    n = len(times)
    nbins = int(np.ceil((lag_stop - lag_start) / bin_width))
    inv_bin = nbins / (lag_stop - lag_start) # faster than dividing at every iteration

    n_samples = np.zeros(nproc,dtype=np.int64)
    ccg = np.zeros((nproc,nproc,nbins),dtype=np.int64)
    for i in range(n):
        ti = times[i]
        pi = proc[i]
        n_samples[pi] += 1

        # find first event that can contribute, starting from next event
        j0 = i + 1
        while j0 < n and times[j0] - ti < lag_start:
            j0 += 1
        j = j0

        max_dt = max(lag_stop,-lag_start) # termination condition
        while j < n:
            dt = times[j] - ti
            if dt >= max_dt:
                break
            pj = proc[j]

            # positive lag contribution
            if lag_start <= dt < lag_stop:
                b = int((dt - lag_start) * inv_bin)
                if 0 <= b < nbins:
                    ccg[pi,pj,b] += 1

            # negative lag contribution (swap reference and target)
            neg_dt = -dt
            if lag_start <= neg_dt < lag_stop:
                b = int((neg_dt - lag_start) * inv_bin)
                if 0 <= b < nbins:
                    ccg[pj,pi,b] += 1

            j += 1

    return ccg, n_samples


def CCG(samples, bin:float=None, limits:tuple[float,float]=None, fast:bool=None, norm:Literal['rate','count','density']=None):
    '''
    compute cross-correlograms from time stamps generated by point processes

    arguments:
        samples    (:,) float | (:,2) float, every row is either [sample time (s)] or [sample time (s), process id]; if 1d, interpreted as (n,1)
        bin        float = 0.05 s, time bin to count samples
        limits     (2) float = [-0.5,0.5] (s), defines a window around time stamps, divided into time bins to compute CCG
        fast       bool = False, if True, 'samples' must be time sorted to save computation time
        norm       str = {'rate','count','density'}, determines unit of measurement of 'ccg'

    output:
        ccg        DESCRIBE
        t          DESCRIBE
    '''

    if bin is None: bin = 0.05
    if limits is None: limits = (-0.5,0.5)
    if norm is None: norm = 'rate'

    samples = np.asarray(samples)
    if samples.ndim == 1:
        times = samples.astype(np.float64)
        id = np.zeros(len(times),dtype=np.int64)
    elif samples.shape[1] == 2:
        times = samples[:,0].astype(np.float64)
        id = samples[:,1].astype(np.int64)
    else:
        raise ValueError("'samples' must be (n,) or (n,2)")
    nproc = id.max() + 1

    # sort by time
    if not fast:
        order = np.argsort(times)
        times = times[order]
        id = id[order]

    ccg, n_samples = _ccg_numba(times, id, nproc, float(bin), float(limits[0]), float(limits[1]))
    edges = np.linspace(limits[0],limits[1],ccg.shape[2]+1)
    t = (edges[1:] + edges[:-1]) / 2

    n_samples[n_samples == 0] = 1  # avoid division by 0, CCG should be zero anyways for them
    if norm == 'rate':
        ccg = ccg.astype(float) / (n_samples[:,None,None] * bin)
    elif norm == 'density':
        ccg = ccg.astype(float) / (n_samples[:,None,None] * n_samples[None,:,None] * bin)

    return ccg, t


def avalanchesFromProfile(x, threshold:float, time_step:float, t0:float=0):
    '''
    compute avalanches' sizes and [start,stop] intervals from a time series

    arguments:
        x            (:) float, time series uniformly sampled in time
        threshold    float in [0,100] (%), percentile of x used as a threshold
        time_step    float, time distance (s) between two consecutive elements of x
        t0           float = 0 (s), time corresponding to first element of x

    output:
        sizes        (n) float, avalanche sizes
        intervals    (n,2) float, each row is an avalanche's [start, stop] interval (s)
        size_t       (m) float, size over time, in which every avalanche is separated by a 0
    '''

    x = np.asarray(x)

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


def PDF(x, mode:Literal['normal','log','polar']=None, method:Literal['kde','discrete']=None, bandwidth:float|str=None, eps:float=None, n_points:int=None,
        bins=None, norm:Literal['density','max']=None):
    """estimate probability density function (PDF) from data

    arguments:
        x            (:,) numeric, values drawn from a stochastic variable X, used to estimate its PDF
        mode         str = {'normal','log','polar'}, DESCRIBE
        method       str = {'kde','discrete'}, DESCRIBE (only for 'normal' mode)
        bandwidth    float | str = 'scott', bandwidth for gaussian kernel
        eps          float = 1e-12, small value used to avoid log(0)
        n_points     int = 50, number of points used to evaluate PDF, ignored if 'bins' are provided
        bins         (:,) float = None, bin edges, if None, bins are linearly spaced between min and max of x
        norm         str = {'density','max','cdf'}, normalization mode, 'density' computes PDF, 'max' normalizes its maximum to 1,
                     'cdf' computes cumulative density function (CDF)

    output:
        grid         (n_points,) float, values of X for which the PDF was evalueated
        density      (n_points,) float, estimated PDF
    """

    if mode is None: mode = 'normal'
    if method is None: method = 'kde'
    if bandwidth is None: bandwidth = 0.05 if mode == 'polar' else 'scott'
    if eps is None: eps = 1e-12
    if n_points is None and bins is None: n_points = 50
    if norm is None: norm = 'density'

    # validate 'x'
    x = np.asarray(x)
    x = x[~np.isnan(x)] # always ravels input: loosing a capability of gaussian_kde?
    if x.size < 2:
        return np.empty(0), np.empty(0)

    match mode:
        # 1. real-valued data using gaussian kernel density estimator
        case 'normal':
            if method == 'kde':
                if n_points is None:
                    grid = bins
                else:
                    grid = np.linspace(x.min(),x.max(),n_points) # linear grid
                kde = sp.stats.gaussian_kde(x,bw_method=bandwidth)
                density = kde(grid)
            else:
                x_unique = np.unique(x)
                dx = min(np.diff(x_unique))
                n_points = int(np.round((x_unique[-1] - x_unique[0]) / dx)) + 1
                if n_points > 1000:
                    raise ValueError('x should be discrete')
                grid = np.linspace(x_unique[0]-dx/2,x_unique[-1]+dx/2,n_points+1) # linear grid
                density, _ = np.histogram(x,bins=grid,density=True)
                grid = (grid[1:] + grid[:-1]) / 2
            if norm == 'max':
                density /= density.max()
            elif norm == 'cdf':
                density = sp.integrate.cumulative_trapezoid(density,grid,initial=0)

        # 2. log-transformed data using gaussian kernel density estimator
        case 'log':
            if x.min() <= 0:
                raise ValueError('log-transforming data requires positive values')
            x = np.log(x+eps) # log-transform data
            if n_points is None:
                grid = bins
            else:
                grid = np.linspace(x.min(),x.max(),n_points) # linear grid in log-space
            jacobian = np.exp(grid) # jacobian term to transform density back to linear
            kde = sp.stats.gaussian_kde(x,bw_method=bandwidth)
            density = kde(grid) / jacobian
            if norm == 'max':
                density /= density.max()

        # 3. circular data using von-Mises kernel density estimator
        case 'polar':
            x = np.mod(x,2*np.pi)
            if n_points is None:
                grid = bins
            else:
                grid = np.linspace(0,2*np.pi,n_points) # linear grid in [0,2*pi]
            density = np.zeros_like(grid)
            for theta in x:
                density += spst.vonmises.pdf(grid-theta,1/bandwidth)
            if norm == 'max':
                density /= density.max()
            else:
                density /= len(x)
            grid = np.concatenate((grid,grid+2*np.pi))
            density = np.concatenate((density, density))

    return grid, density


def cellAssembliesICA(spikes, window:float=None, when=None, drop_mix:bool=False):
   # detect assemblies from spike trains with PCA + ICA

   try:
       import sklearn.decomposition as skdc
       import skimage.filters as skif
   except ImportError as e:
       raise ImportError('cellAssembliesICA requires scikit-learn, did you do: pip install "fmatoolbox[assemblies]" ?') from e

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

   try:
       import joblib
   except ImportError as e:
       raise ImportError('reactivationStrength requires joblib, did you do: pip install "fmatoolbox[assemblies]" ?') from e

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

def MCpValue(surrogate,observed,alternative='two-sided'):
    """ compute Monte Carlo p-values comparing observed statistics to surrogate distributions

    arguments:
        surrogate      (s,f,...) float, surrogate statistics; s: n of surrogates, f: n of features
        observed       (f,...) float, observed statistics, must have shape equal to surrogate.shape[1:]
        alternative    str = {"two-sided", "greater", "less"}, test direction

    output:
        pvals          (f,) float, Monte Carlo p-values
    """

    surrogate = np.asarray(surrogate)
    observed = np.asarray(observed)
    if np.any(surrogate.shape[1:] != observed.shape):
        raise ValueError("'surrogate' must have the same shape of 'observed', except for the first dimension")

    if alternative == "greater":
        count = np.sum(surrogate >= observed, axis=0)

    elif alternative == "less":
        count = np.sum(surrogate <= observed, axis=0)

    elif alternative == "two-sided":
        greater = np.sum(surrogate >= observed, axis=0)
        less = np.sum(surrogate <= observed, axis=0)
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
       reject           (n,) bool, optional, true for hypothesis that can be rejected
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