"""Specialized statistics routines"""

import fmatoolbox.general
import numpy as np
import scipy as sp
import statsmodels.stats.multitest
from typing import Literal


def MCpValue(surrogate,observed,alternative='two-sided'):
    """compute Monte Carlo p-values comparing observed statistics to surrogate distributions

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
    """Holm-Bonferroni correction for multiple tests

    arguments:
        pvals            ndarray[float], p values, NaNs are ignored in the correction procedure and propagated in output
        alpha            float = 0.05, significance level, must be in [0,1]
        return_reject    bool = False, return also rejection decisions

    output:
        corrected        (n,) float, adjusted p values
        reject           (n,) bool, optional, true for hypothesis that can be rejected
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


def maxStatisticTest(data, surrogate, statistic=None, group=None, alpha:float=0.05, alternative:str='two-sided'):
   """conduct a max statistic test over time, assessing in which time points the null hypothesis about a statistic across sessions can be rejected

   arguments:
       data           (sessions, times) float
       surrogate      (sessions, times, surrogates) float
       group          (sessions,) int, grouping variable used to aggregate sessions, the statistic is computed per group and then again over groups
       alpha          float = 0.05, significance level, must be in [0,1]
       alternative    str = {'two-sided','grater','less'}, test direction, defines the null hypothesis
   """

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


def clusterPermutationTest(data1, data2=None, paired:bool=False, n_perm:int=1000, alpha:float=0.05, tail:Literal['both','right','left']='both',
                            cluster_stat:Literal['size','mass']='size'):
    """perform a cluster-based permutation test on time series data, either to test one sample against zero, or to compare
    two samples, correcting for multiple comparisons while accounting for temporal correlations

    Args:
        data1:           first group's data, shape (n_samples1, times), e.g., z-scored PETH
        data2:           optional, either:
                         - None, to test `data1` against zero
                         - a scalar float, to test `data1-data2` against zero
                         - a (n_samples2, times) float array, second group's data, must have the same number of columns as `data1`, and,
                           if `paired` is True, the same number of rows too
        paired:          if True, run a paired test, ignored if `data2` is None or scalar, defaults to False
        n_perm:          number of permutations used to build the null distribution of the maximum cluster statistic, defaults to 1000
        alpha:           significance level, used both to threshold time points into clusters and to test clusters' significance, defaults to 0.05
        tail:            test direction, one of 'both', 'right', 'left'
        cluster_stat:    statistic used to quantify a cluster, one of 'size' (number of time points, more robust when a cluster's effect
                         is driven by a few extreme values) or 'mass' (sum of `abs(stat)` across its time points)

    Returns:
        dict with fields:
            stat:           observed t-statistic at every time point, shape (times,)
            clusters:       cluster id at every time point, 0 = not part of any cluster, as returned by ``scipy.ndimage.label``, shape (times,)
            cluster_stat:   observed `cluster_stat` statistic for every cluster, in the same order as cluster ids, shape (n_clusters,)
            p_cluster:      monte carlo p value for every cluster, shape (n_clusters,)
            sig_mask:       True at time points belonging to a cluster with `p_cluster` < `alpha`, shape (times,)
            threshold:      critical t value used to threshold time points into clusters, varies over time with the number of
                            non-NaN observations, shape (times,)

    Note:
        a null distribution for the maximum cluster statistic is built by permutation (sign-flip if 'paired', else label-shuffle)
        and used to assign every observed cluster a corrected p value, following Maris & Oostenveld (2007)
    """

    # validate input
    if tail not in ('both','right','left'):
        raise ValueError("'tail' must be 'both', 'right' or 'left'")
    if cluster_stat not in ('size','mass'):
        raise ValueError("'cluster_stat' must be 'size' or 'mass'")

    data1 = np.array(data1, dtype=float, ndmin=2)
    data2 = np.array(0., ndmin=2) if data2 is None else np.array(data2, dtype=float, ndmin=2)
    if data2.size == 1:
        # one-sample case
        paired = True
        data2 = np.full_like(data1, data2.item())
    n1, T = data1.shape
    n2, T2 = data2.shape
    if T != T2:
        raise ValueError("'data1' and 'data2' must have the same number of columns")
    if paired and n1 != n2:
        raise ValueError("a paired test requires 'data1' and 'data2' to have the same number of rows")

    # 1. define functions used to compute statistics

    # statitic computed per time point
    def statistic(x,y=None):
        if paired:
            # one-sample t-test, 'x' is a difference
            n = np.sum(~np.isnan(x), axis=0)
            mu = np.nanmean(x, axis=0)
            sigma = np.nanstd(x, axis=0, ddof=1)
            stat = mu / (sigma / np.sqrt(n))
            df = n - 1
        else:
            # two-samples t-test
            # NOTE: could replace Student's t-statistic with Welch's t-statistic to drop equal variance assumption
            n1t = np.sum(~np.isnan(x), axis=0);    n2t = np.sum(~np.isnan(y), axis=0)
            mu1 = np.nanmean(x, axis=0);           mu2 = np.nanmean(y, axis=0)
            s1 = np.nanstd(x, axis=0, ddof=1);     s2 = np.nanstd(y, axis=0, ddof=1)
            pooled = np.sqrt(((n1t-1) * s1 ** 2 + (n2t-1) * s2 ** 2) / (n1t+n2t-2))
            stat = (mu1 - mu2) / (pooled * np.sqrt(1/n1t + 1/n2t))
            df = n1t + n2t - 2
        return stat, df

    # cluster statistics
    def cluster(s,thr):
        if tail == 'both':
            sig = np.abs(s) > thr
        elif tail == 'right':
            sig = s > thr
        else: # 'left'
            sig = s < -thr
        # find clusters
        clusters, n_clusters = sp.ndimage.label(sig)
        if n_clusters:
            if cluster_stat == 'size':
                cluster_stats = np.bincount(clusters)[1:]  # exclude background label 0
            else:  # 'mass'
                cluster_stats = np.bincount(clusters, weights=np.abs(s))[1:]
        else:
            cluster_stats = np.empty(0)
        return clusters, n_clusters, cluster_stats

    # prepare data
    if paired:
        data_diff = data1 - data2 # run test on the difference
    else:
        combined = np.concatenate((data1,data2), axis=0) # used for permutations later

    # 2. observed statistic
    stat, df = statistic(data_diff) if paired else statistic(data1,data2)
    # threshold for clustering, 'thresh' and 'df' vary over time with the number of non-NaN observations
    thresh = sp.stats.t.ppf(1-alpha/2,df) if tail == 'both' else sp.stats.t.ppf(1-alpha,df)
    clusters, n_clusters, cluster_stats = cluster(stat,thresh)

    # 3. surrogate statistic: build null distribution of the maximum cluster statistic
    rng = np.random.default_rng()
    max_cluster_perm = np.empty(n_perm)
    for i in range(n_perm):
        if paired:
            # sign-flip permutation
            signs = rng.choice((-1,1),size=(n1,1)) # random sign per row, applied to its entire time series
            perm_data = data_diff * signs
            stat_perm, _ = statistic(perm_data)
        else:
            # label-shuffle permutation
            perm_idx = rng.permutation(n1 + n2)
            stat_perm, _ = statistic(combined[perm_idx[:n1]], combined[perm_idx[n1:]])
        # surrogate cluster statistics
        clust_perm, n_cp, perm_cluster_stats = cluster(stat_perm,thresh)
        max_cluster_perm[i] = perm_cluster_stats.max() if n_cp else 0

    # 4. cluster p-values and significance mask
    if n_clusters:
        p_cluster = MCpValue(np.tile(max_cluster_perm,(n_clusters,1)).T,cluster_stats,'greater')
        sig_mask = np.isin(clusters, np.where(p_cluster < alpha)[0] + 1)
    else:
        p_cluster = np.empty(0)
        sig_mask = np.zeros(T,dtype=bool)

    return {'stat':stat, 'clusters':clusters, 'cluster_stat':cluster_stats, 'p_cluster':p_cluster, 'sig_mask':sig_mask, 'threshold':thresh}


def hierarchicalBootstrap(x, groupx, y=None, groupy=None, paired=None, n_iter=1000):
    """hierarchical bootstrap for nested grouped observations
    at each level groups are sampled with replacement, and so are observations at the lowest level

    Args:
        x:         data for condition X, shape (n_samples_x, n_features)
        groupx:    hierarchical grouping variables for X, shape (n_samples_x, n_groups), columns ordered from lowest to highest level,
                   e.g. groupx[:,0] = trial, groupx[:,1] = subject, giving the hierarchy: subject -> trial -> observations
        y:         optional data for condition Y, shape (n_samples_y, n_features), if None only X is bootstrapped
        groupy:    optional grouping variables corresponding to `y`, shape (n_samples_y, n_groups)
        paired:    number of paired levels treated as a repeated-measures comparison, counting from the top; groups at those levels are sampled
                   jointly so they occur in both conditions, requiring common group identifiers in `groupx` and `groupy`, defaults to 0
        n_iter:    number of bootstrap iterations, defaults to 1000

    Returns:
        means:     bootstrap distribution of the data mean, ignoring nans, shape depends on whether `y` is:
                   - None, shape (n_iter, n_features) giving the bootstrap distribution of the mean of X
                   - given, shape (n_iter, n_features, 3), with the last dimension being X, Y, X - Y
        p_value:   p values for the following tests: "mean of X is 0" and, optionally, "mean of Y is 0", "mean of X - Y is 0";
                   shape is either (n_features,) or (n_features,3)
        ci:        confidence intervals for `means`, shape is either (2, n_features) or (2, n_features, 3)
    """

    # validate input
    x = np.asarray(x, dtype=float)
    groupx = np.asarray(groupx)
    if x.ndim == 1:         x = x[:,None]
    if groupx.ndim == 1:    groupx = groupx[:,None]
    n_levels = groupx.shape[1]
    if x.shape[0] != groupx.shape[0]:
        raise ValueError("'x' and 'groupx' must have the same number of samples (rows)")
    if y is not None:
        if groupy is None:
            raise ValueError("'groupy' must be given when 'y' is given")
        y = np.asarray(y, dtype=float)
        groupy = np.asarray(groupy)
        if y.ndim == 1:         y = y[:,None]
        if groupy.ndim == 1:    groupy = groupy[:,None]
        if y.shape[0] != groupy.shape[0]:
            raise ValueError("'y' and 'groupy' must have the same number of samples (rows)")
        if x.shape[1] != y.shape[1]:
            raise ValueError("'x' and 'y' must have the same number of features (columns)")
        if n_levels != groupy.shape[1]:
            raise ValueError("'groupx' and 'groupy' must have the same number of hierarchical levels (columns)")
        if paired is None: paired = 0
        if int(paired) != paired or not (0 <= paired <= n_levels):
            raise ValueError(f"'paired' must be an integer between 0 and {n_levels}")
        first_paired = n_levels - int(paired) # levels >= first_paired are paired

    # define functions

    rng = np.random.default_rng()
    resample = lambda x : rng.choice(x, size=len(x), replace=True)

    def recursive_sample(indices,groups,level):
        if level < 0:
            return resample(indices)
        group_ids = np.unique(groups[indices,level]) # groups at this level
        sampled_groups = resample(group_ids)
        return np.concatenate([recursive_sample(indices[groups[indices,level] == group_id], groups, level-1)
            for group_id in sampled_groups])

    def recursive_sample2(idx_x,idx_y,groupx,groupy,level,paired):
        if level < 0:
            return resample(idx_x), resample(idx_y)
        if level >= paired:
            # sample groups jointly
            group_ids = np.intersect1d(np.unique(groupx[idx_x,level]), np.unique(groupy[idx_y,level]))
            if len(group_ids) == 0:
                raise ValueError(f"'x' and 'y' share no group at a paired hierarchy level ({level})")
            sampled_groups = resample(group_ids)
            # for each sampled group, recurse to lower level
            out_x, out_y = [], []
            for group_id in sampled_groups:
                ix = idx_x[groupx[idx_x,level] == group_id]
                iy = idx_y[groupy[idx_y,level] == group_id]
                rx, ry = recursive_sample2(ix, iy, groupx, groupy, level-1, paired)
                out_x.append(rx)
                out_y.append(ry)
            return np.concatenate(out_x), np.concatenate(out_y)
        # if unpaired, from this level onward resort to independent sampling of 'x' and 'y'
        return recursive_sample(idx_x,groupx,level), recursive_sample(idx_y,groupy,level)

    def summarize(boot):
        return MCpValue(boot,np.zeros(boot.shape[1:])), np.nanpercentile(boot, [2.5,97.5], axis=0)

    # 1. single-condition bootstrap
    if y is None:
        boot_means = np.full((n_iter,x.shape[1]), np.nan)
        for i in range(n_iter):
            indices = recursive_sample(np.arange(groupx.shape[0]), groupx, groupx.shape[1]-1)
            boot_means[i] = np.nanmean(x[indices], axis=0)
        p_value, ci = summarize(boot_means) # (n_features,), (2, n_features)
        return boot_means, p_value, ci

    # 2. two-conditions bootstrap
    boot_x = np.full((n_iter,x.shape[1]), np.nan)
    boot_y = np.full((n_iter,y.shape[1]), np.nan)
    for i in range(n_iter):
        idx_x, idx_y = recursive_sample2(np.arange(groupx.shape[0]), np.arange(groupy.shape[0]), groupx, groupy, groupx.shape[1]-1, first_paired)
        boot_x[i] = np.nanmean(x[idx_x], axis=0)
        boot_y[i] = np.nanmean(y[idx_y], axis=0)
    boot = np.stack([boot_x,boot_y,boot_x-boot_y], axis=-1) # (n_iter, n_features, 3)
    p_value, ci = summarize(boot) # (n_features, 3), (2, n_features, 3)
    return boot, p_value, ci