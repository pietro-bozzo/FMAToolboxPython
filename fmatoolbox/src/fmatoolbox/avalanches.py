import numpy as np


def avalanchesFromProfile(x, threshold:float, time_step:float, t0:float=None):
    """compute avalanches' sizes and [start,stop] intervals from a time series

    Args:
        x:            (:) float, time series uniformly sampled in time
        threshold:    percentile of x used as a threshold, must be in [0,100]
        time_step:    time distance (s) between two consecutive elements of x
        t0:           time (s) corresponding to first element of x, defaults to 0

    Returns:
        sizes:        (n) float, avalanche sizes
        intervals:    (n,2) float, each row is an avalanche's [start, stop] interval (s)
        size_t:       (m) float, size over time, in which every avalanche is separated by a 0
    """

    x = np.asarray(x)
    if t0 is None: t0 = 0.

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

    # start and stop times
    start = np.where(np.concatenate(([x[0] != 0], (x[1:] != 0) & (x[:-1] == 0))))[0]
    stop = np.where(np.concatenate(((x[1:] == 0) & (x[:-1] != 0), [x[-1] != 0])))[0] + 1
    intervals = np.stack((start, stop), 1) * time_step + t0 - time_step / 2

    return sizes, intervals, size_t


def avalTransitionMatrix(x, threshold, min_length:int=None, time_step:float=None, t0:float=None):
    """estimate P[i,j] = P(x[t+1,i] = True | x[t,j] = True) per avalanche,
    using only transitions where both t and t+1 lie inside the block

    Returns a list of (start, end, P), end exclusive.
    P is (m, m) with NaN where column j is never True at a usable t,
    and None for single-row blocks (no transitions).
    """

    if min_length is None: min_length = 2
    if t0 is None: t0 = 0.

    n_features = x.shape[1]
    threshold = np.percentile(x, threshold, axis=0) # convert from percentile
    x = x > threshold[None,:]

    # find avalanches
    active = x.any(axis=1)
    d = np.diff(np.concatenate(([0], active.view(np.int8), [0])))
    intervals = np.stack((np.flatnonzero(d == 1), np.flatnonzero(d == -1)), axis=1) # intervals[i]: [start, end (exclusive)]
    valid = np.ravel(np.diff(intervals) >= min_length)
    intervals = intervals[valid]

    ATM = np.full((len(intervals), n_features, n_features), np.nan)
    for i, (s, e) in enumerate(intervals):
        this_x = x[s:e].astype(np.float64)
        prev, nxt = this_x[:-1], this_x[1:]
        counts = nxt.T @ prev # counts[i,j] = x[t,j] & x[t+1,i]
        denom = prev.sum(axis=0) # x[t,j] over usable t
        with np.errstate(invalid="ignore", divide="ignore"):
            ATM[i] = counts / denom[None,:] # divides column j by denom[j]

    # convert to seconds
    if time_step is not None:
        intervals = intervals.astype(float) * time_step
        intervals += t0 - time_step / 2

    return ATM, intervals


def powerlaw_fit(data, xmin=None, xmax=None, drop_unique=False, verbose=False):
    """fit a discrete power law via Clauset et al. 2009 (MLE)"""

    data = np.round(data).astype(int)
    data = data[data >= 1]
    data_unique, counts = np.unique(data, return_counts=True)
    verbose and print(len(data), "avalanches for power-law fit", len(data_unique), "unique values")
    result = {key: np.nan for key in ('tau','sigma','xmin','xmax','r','p','tail_prop','ks','pdf')}

    # validate input
    if len(data_unique) < 2 or len(data) < 20:
        verbose and print("DATA TOO SMALL OR NOT ENOUGH UNIQUE VALUES FOR POWER-LAW FIT")
        return result
    most_common_freq = np.max(counts) / len(data)
    if most_common_freq > 0.90:
        verbose and print("MOST COMMON VALUE TOO PREVALENT FOR POWER-LAW FIT")
        return result
    if drop_unique:
        data = data[np.isin(data, data_unique[counts > 1])] # drop unique data points

    # empirical PDF of the data (log-binned by default — standard for power-law tails)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            bin_edges, probability = powerlaw.pdf(data, linear_bins=False)
            center = np.sqrt(bin_edges[:-1] * bin_edges[1:]) # bin center: geometric mean of edges
            result['pdf'] = np.stack((center,probability), axis=1)
        except Exception:
            pass

    if xmax is not None:
        xmax = int(np.percentile(data,xmax)) # apply percentile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", OptimizeWarning)

        try:
            fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax, discrete=True, verbose=False)
        except Exception:
            return result
        if np.isnan(fit.power_law.alpha):
            return result

        result['tau'] = fit.power_law.alpha
        result['sigma'] = fit.power_law.sigma
        result['xmin'] = fit.xmin
        result['xmax'] = fit.xmax if fit.xmax is not None else np.max(data_unique[-1])
        result['tail_prop'] = fit.n_tail / len(data)

        # compare power law against exponential
        result['r'], result['p'] = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)

        # KS statistics, unused for now
        tail = data[(data >= result['xmin']) & (data <= result['xmax'])]
        theoretical = fit.power_law.cdf(np.sort(tail))
        empirical   = np.arange(1,len(tail)+1) / len(tail)
        result['ks'] = float(np.max(np.abs(empirical - theoretical)))

    return result


def critical_exponents(sizes, intervals, bin_size:float, xmin:float=None, xmax:float=None, verbose:bool=None):
    "estimate critical exponents and distance from criticality"

    exponents = {}; fit = {}

    # sizes
    size_fit = powerlaw_fit(sizes, xmin=xmin, xmax=xmax, verbose=verbose)
    exponents['tau_s'] = size_fit['tau']
    size_fit.pop('tau')
    fit['S'] = size_fit

    # durations
    durations = np.round(np.diff(intervals).ravel() / bin_size)
    dur_fit = powerlaw_fit(durations, xmin=xmin, xmax=xmax, verbose=verbose)
    exponents['tau_d'] = dur_fit['tau']
    dur_fit.pop('tau')
    fit['D'] = dur_fit

    def _gamma_from_area(sizes, durations):
        """estimate γ via OLS regression of log <S|D> on log D"""
        sizes     = np.asarray(sizes,     dtype=float)
        valid = (sizes >= 1) & (durations >= 1)
        S = sizes[valid]; D = durations[valid]

        unique_D = np.unique(D)
        if len(unique_D) < 4:
            return np.nan

        mean_S = np.array([S[D == d].mean() for d in unique_D])
        log_D  = np.log(unique_D)
        log_S  = np.log(mean_S)

        keep = np.isfinite(log_D) & np.isfinite(log_S)
        if keep.sum() < 4:
            return np.nan

        gamma, _ = np.polyfit(log_D[keep], log_S[keep], 1)
        return float(gamma)

    # DCC
    exponents['gamma_th'] = (exponents['tau_d'] - 1) / (exponents['tau_s'] - 1) # theoretical γ from scaling relation of τ exponents
    valid = (sizes >= fit['S']['xmin']) & (sizes <= fit['S']['xmax']) & (durations >= fit['D']['xmin']) & (durations <= fit['D']['xmax'])
    exponents['gamma_emp'] = _gamma_from_area(sizes_this[valid], durations[valid]) # empirical γ from <S> ~ D^γ
    exponents['dcc'] = abs(exponents['gamma_th'] - exponents['gamma_emp']) # DCC from scaling relation

    return exponents, fit