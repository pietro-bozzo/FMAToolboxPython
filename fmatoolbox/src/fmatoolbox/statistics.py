''' Specialized statistics routines '''

import fmatoolbox.general
import numpy as np
import scipy as sp
import statsmodels.stats.multitest
from typing import Literal, Callable


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

    arguments:
        data1           (n1,T) float, first group's data, e.g., z-scored PETH
        data2           optional, either:
                        - None, to test data1 against zero
                        - a scalar float, to test data1-data2 against zero
                        - (n2,T) float, second group's data, must have the same number of columns as 'data1', and, if 'paired'
                        is True, the same number of rows too
        paired          bool = False, if True, run a paired test (ignored if data2 is None or scalar)
        n_perm          int = 1000, number of permutations used to build the null distribution of the maximum cluster statistic
        alpha           float = 0.05, significance level, used both to threshold time points into clusters and to test clusters' significance
        tail            str = {'both','right','left'}, test direction
        cluster_stat    str = {'size','mass'}, statistic used to quantify a cluster: 'size' = number of time points (more robust when
                        a cluster's effect is driven by a few extreme values), 'mass' = sum of abs('stat') across its time points

    output:
        stats    dict with fields:
                 'stat'          (T,) float, observed t-statistic at every time point
                 'clusters'      (T,) int, cluster id at every time point (0 = not part of any cluster), as returned by `scipy.ndimage.label`
                 'cluster_stat'  (n_clusters,) float, observed 'cluster_stat' statistic for every cluster, in the same order as cluster ids
                 'p_cluster'     (n_clusters,) float, Monte Carlo p value for every cluster
                 'sig_mask'      (T,) bool, True at time points belonging to a cluster with 'p_cluster' < 'alpha'
                 'threshold'     (T,) float, critical t value used to threshold time points into clusters (varies over time with the number
                                 of non-NaN observations)

    notes:
        t points thresholded at critical thrs for 'alpha' (uncorrected) to form clusters; a null distribution for the maximum
        cluster statistic is then built by permutation (sign-flip if 'paired', else label-shuffle) and used to assign every observed
        cluster a corrected p value, following Maris & Oostenveld (2007)
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


def hierarchicalBootsrap(x, groupx, y=None, groupy=None, paired=False, n_iter=1000):
    """hierarchical bootstrap for nested/grouped observations

    arguments:
    x : np.ndarray, shape (n_samples_x, n_features)
        Data for condition X.

    groupx : np.ndarray, shape (n_samples_x, n_groups)
        Hierarchical grouping variables for X.

        Columns should be ordered from lowest to highest level.
        For example:

            groupx[:, 0] = trial
            groupx[:, 1] = subject

        Thus the hierarchy is:

            subject -> trial -> observations

    y : np.ndarray, optional, shape (n_samples_y, n_features)
        Data for condition Y. If None, only X is bootstrapped.

    groupy : np.ndarray, optional, shape (n_samples_y, n_groups)
        Grouping variables corresponding to y.

    paired : (n_groups,) bool, default=False
        If True, X and Y are treated as a paired/repeated-measures comparison. Groups at the highest common level are sampled
        jointly so that the same subjects occur in both conditions. This requires a common highest-level group identifier in groupx and groupy.

    n_iter : int, default=1000
        Number of bootstrap iterations.

    Returns
    -------
    result : np.ndarray
        If y is None:

            shape (n_iter, n_features)

            Bootstrap distribution of the mean of X.

        If y is given:

            shape (n_iter, 3, n_features)

            result[:, 0, :] = bootstrap means of X
            result[:, 1, :] = bootstrap means of Y
            result[:, 2, :] = X - Y

    Notes
    -----
    The bootstrap is hierarchical:

        highest-level group
            -> next grouping level
                -> ...
                    -> individual observations

    At each level, groups are sampled with replacement, and at the
    lowest level individual observations are sampled with replacement.

    NaNs are ignored when calculating means.
    """

    # validate input
    x = np.asarray(x, dtype=float)
    groupx = np.asarray(groupx)
    if x.ndim == 1:
        x = x[:,None]
    if groupx.ndim == 1:
        groupx = groupx[:,None]
    if x.shape[0] != groupx.shape[0]:
        raise ValueError("'x' and 'groupx' must have the same number of samples")
    if y is not None:
        y = np.asarray(y, dtype=float)
        groupy = np.asarray(groupy)
        if y.ndim == 1:
            y = y[:,None]
        if groupy.ndim == 1:
            groupy = groupy[:,None]
        if y.shape[0] != groupy.shape[0]:
            raise ValueError("'y' and 'groupy' must have the same number of samples")
        if x.shape[1] != y.shape[1]:
            raise ValueError("'x' and 'y' must have the same number of features")
        if groupx.shape[1] != groupy.shape[1]:
            raise ValueError("'groupx' and 'groupy' must have the same number of hierarchical levels")

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
        if paired[level]:
            # sample groups jointly
            ids = np.intersect1d(np.unique(groupx[idx_x,level]), np.unique(groupy[idx_y,level]))
            if len(ids) == 0:
                raise ValueError(f"when 'paired' is True, 'x' and 'y' must share at least one group at that hiearchy level ({level})")
            sampled = resample(ids)
            # for each sampled group, recurse to lower level
            out_x, out_y = [], []
            for group_id in sampled:
                ix = idx_x[groupx[idx_x,level] == group_id]
                iy = idx_y[groupy[idx_y,level] == group_id]
                rx, ry = recursive_sample2(ix, iy, groupx, groupy, level-1, paired)
                out_x.append(rx)
                out_y.append(ry)
        else:
            # from this level, resort to independent sampling of 'x' and 'y' CAN REPLACE WITH TWO SIMPLE CALLS to recursive_sample(idx_x,groupx,level) ??
            ids_x = np.unique(groupx[idx_x,level])
            sampled_x = resample(ids_x)
            out_x = [recursive_sample(idx_x[groupx[idx_x,level] == group_id], groupx, level-1) for group_id in sampled_x]
            ids_y = np.unique(groupy[idx_y,level])
            sampled_y = resample(ids_y)
            out_y = [recursive_sample(idx_y[groupy[idx_y,level] == group_id], groupy, level-1) for group_id in sampled_y]
        return np.concatenate(out_x), np.concatenate(out_y)

    # 1. single-condition bootstrap
    if y is None:
        boot_means = np.full((n_iter,x.shape[1]), np.nan)
        for i in range(n_iter):
            indices = recursive_sample(np.arange(groupx.shape[0]), groupx, groupx.shape[1]-1)
            boot_means[i] = np.nanmean(x[indices], axis=0)
        ci = np.nanpercentile(boot_means, [2.5,97.5], axis=0)
        p_value = (ci[0] > 0) | (ci[1] < 0)
        return boot_means, p_value, ci

    # 2. two-conditions bootstrap
    boot_x = np.full((n_iter,x.shape[1]), np.nan)
    boot_y = np.full((n_iter,y.shape[1]), np.nan)
    for i in range(n_iter):
        idx_x, idx_y = recursive_sample2(np.arange(groupx.shape[0]), np.arange(groupy.shape[0]), groupx, groupy, groupx.shape[1]-1, paired)
        boot_x[i] = np.nanmean(x[idx_x], axis=0)
        boot_y[i] = np.nanmean(y[idx_y], axis=0)
    boot_diff = boot_x - boot_y

    return np.stack([boot_x,boot_y,boot_diff], axis=0)