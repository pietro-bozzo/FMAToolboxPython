"""Specialized statistics routines"""

import numpy as np
import scipy as sp
import statsmodels.stats.multitest
from typing import Literal, Iterable


def MCpValue(surrogate,observed,alternative:Literal['two-sided','greater','less']=None):
    """compute Monte Carlo p-values comparing observed statistics to surrogate distributions

    Args:
        surrogate:     surrogate statistics, first dimension corresponds to surrogate replicates
        observed:      observed statistics, must have shape equal to `surrogate.shape[1:]`
        alternative:   test direction

    Returns:
        pvals:         Monte Carlo p-values, same shape as `observed`
    """

    if alternative is None: alternative = 'two-sided'
    surrogate = np.asarray(surrogate)
    observed = np.asarray(observed)
    if surrogate.shape[1:] != observed.shape:
        raise ValueError("'surrogate' must have the same shape of 'observed', except for the first dimension")

    if alternative == "greater":
        count = np.sum(surrogate >= observed, axis=0)
    elif alternative == "less":
        count = np.sum(surrogate <= observed, axis=0)
    elif alternative == "two-sided":
        greater = np.sum(surrogate >= observed, axis=0)
        less = np.sum(surrogate <= observed, axis=0)
        count = 2 * (np.minimum(greater,less) + 1) - 1 # so that later (count+1) = 2*(min+1)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    n = np.sum(~np.isnan(surrogate), axis=0)
    pvals = (count + 1) / (n + 1) # +1 implement finite-sample Monte Carlo correction (Phipson & Smyth 2010, CHECK REF)
    pvals = np.where(n==0, np.nan, pvals) # all-NaN feature have no p-value

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


def hierarchicalBootstrap(data, groups, skip:Iterable[int]=None, n_iter:int=None, rng=None):
    """hierarchical bootstrap estimate of the distribution of the mean of nested grouped observations
    note: this function is a wrapper around utilities of ``hierarch.resampling``, see Kulkarni et al. 2022 https://doi.org/10.1371/journal.pcbi.1010061

    Args:
        data:      tuple containing datasets for different conditions, each of shape (n_samples_i, n_features_i)
        groups:    tuple containing hierarchical grouping variables for each condition, each of shape (n_samples_i, n_levels_i),
                   columns are ordered from outer to deepest level, e.g. groupx[:,0] = subject, groupx[:,1] = trial,
                   giving the hierarchy: subject -> trial -> observations
        skip:      levels that should not be resampled, defaults to ()
        n_iter:    number of bootstrap iterations, defaults to 5000
        rng:       ``np.random.Generator`` or seed, defaults to ``np.random.default_rng()``

    Returns:
        means:     bootstrap distribution of the data mean, ignoring nans, shape depends on whether `y` is:
                   - None, shape (n_iter, n_features) giving the bootstrap distribution of the mean of X
                   - given, shape (n_iter, n_features, 3), with the last dimension being X, Y, X - Y
    """

    try:
       import hierarch
       import pandas as pd
    except ImportError as e:
       raise ImportError('hierarchicalBootstrap requires hierarch, did you do: pip install "fmatoolbox[hierarch]" ?') from e

    # defaults
    if not isinstance(data, tuple):
        data = (data,)
    if not isinstance(groups, tuple):
        groups = (groups,)
    if skip is None: skip = ()
    if n_iter is None: n_iter = 5000
    if rng is None: rng = np.random.default_rng()

    # bootstrap each condition
    means = []
    for i, (d, g) in enumerate(zip(data,groups)):

        # validate shape
        d, g = np.asarray(d), np.asarray(g)
        if len(d) != len(g):
            raise ValueError(f"data[{i}] and groups[{i}] must have the same number of samples (rows)")
        one_d = d.ndim == 1
        if one_d: d = d[:,None]
        if g.ndim == 1: g = g[:,None]

        # handle n-dimensional data
        trailing = d.shape[1:]
        d = d.reshape(len(d),-1) # (samples, prod(trailing))
        keep = ~np.isnan(d).all(axis=1) # drop slices that are entirely NaN
        d, g = d[keep], g[keep]

        # set up design matrix
        df = pd.DataFrame(g, columns=[f"group{k}" for k in range(g.shape[1])])
        df.insert(0, "treatment", 0) # add dummy treatment column
        df["values"] = np.arange(len(d), dtype=float) # row ids, to keep track of reordering
        formula = f"values ~ {'/'.join(c for c in df.columns if c != 'values')}"
        matrix, _ = hierarch.design.design_matrix(df, formula, treatment="treatment")
        X = matrix.drop(columns=["values","treatment"]).to_numpy(float)
        order = matrix["values"].to_numpy(int)

        # resample
        plan = hierarch.resampling.bootstrap_plan(X, skip=skip)
        w = hierarch.resampling.draw_bootstrap_weights_batch(plan, rng=rng, start=0, kind="weights", size=n_iter)
        d = d[order]
        ok = ~np.isnan(d)
        num = w @ np.where(ok, d, 0.0); den = w @ ok.astype(float)
        m = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
        means.append(m.ravel() if one_d else m.reshape((n_iter,) + trailing))

    return means


def hierarchicalBootstrapTest(data, groups=None, names=None, compare=None, which=None, **kwargs):
    """hierarchical bootstrap test for the difference of mean between conditions
    construct a group-aware bootstrap distribution for the sample means and tests the null hypothesis that the mean of the difference between conditions is zero
    note: this function is a wrapper around utilities of ``hierarch.stats``, see Kulkarni et al. 2022 https://doi.org/10.1371/journal.pcbi.1010061

    Args:
        data:      tuple containing datasets for different conditions, each of shape (n_samples_i, n_levels1 + 1)
        groups:    optional tuple containing hierarchical grouping variables for each condition, each of shape (n_samples_i, n_levels2),
                   columns are ordered from outer to deeper level, e.g. groupx[:,0] = subject, groupx[:,1] = trial,
                   giving the hierarchy: subject -> trial -> observations
        names:     optional list of names for each condition, defaults to `range(len(data))`, ignored if data has less than 3 elements
        compare:   test statistic, either "means" (default), "corr", "jackknife_corr", or a Callable
        which:     pairs of datasets to test, when data has more than two elements; if "all" every pairwise comparison will be tested, if a list of lists,
                   each element contains the indeces of a pair, defaults to "all"
        **kwargs:  all extra key-word arguments are passed to the following function, depending on `len(data)`:
                   -1: IMPLEMENT
                   -2: ``hierarch.stats.hypothesis_test``
                   -3: ``hierarch.stats.multi_sample_test``
                   examples include: alternative, skip, bootstraps, permutations, kind, random_state

    Returns:
        p_value:   p values for the following tests, depending on `len(data)`:
                   - 1: to implement
                   - 2: float, p value for the null hypothesis between the two conditions
                   - 3 or more: pandas DataFrame with columns condtion1, condition2, p value, corrected p value
    """

    try:
       import hierarch
       import pandas as pd
    except ImportError as e:
       raise ImportError('hierarchicalBootstrapTest requires hierarch, did you do: pip install "fmatoolbox[hierarch]" ?') from e

    # defaults
    if not isinstance(data, tuple):
        data = (data,)
    if groups is None:
        groups = [np.empty((len(d),0)) for d in data]
    elif not isinstance(groups, tuple):
        groups = (groups,)
    if which is None: which = 'all'
    if compare is None: compare = 'means'

    # convert data to long-format pandas DataFrame
    frames = []
    n_cols = None
    for i, (d, g) in enumerate(zip(data, groups)):

        # validate shape
        d, g = np.asarray(d), np.asarray(g)
        if len(d) != len(g):
            raise ValueError(f"data[{i}] and groups[{i}] must have the same number of samples (rows)")
        if d.ndim == 1: d = d[:,None]
        if g.ndim == 1: g = g[:,None]

        # drop missing values
        valid = ~np.isnan(d[:,-1])
        d = d[valid]
        g = g[valid]

        # separate groups and data
        gcols = np.column_stack((g, d[:,:-1])) if d.shape[1] > 1 else g
        f = pd.DataFrame(gcols, columns=[f"group{k}" for k in range(gcols.shape[1])])
        f.insert(0, "treatment", i) # add treatment column
        f["values"] = d[:,-1].astype(float)
        frames.append(f)

        # validate number of columns
        if n_cols is not None and n_cols != f.shape[1]:
            raise ValueError(f"datasets must have the same number of hierarchy levels (columns)")
        n_cols = f.shape[1]
    df = pd.concat(frames, ignore_index=True)

    # set up design matrix for null-hypothesis testing
    formula = f"values ~ {'/'.join(c for c in df.columns if c != "values")}"
    matrix, treatment_col = hierarch.design.design_matrix(df, formula, treatment='treatment')

    if len(data) == 1:
        def summarize(boot):
            return MCpValue(boot, np.zeros(boot.shape[1:])), np.nanpercentile(boot, [2.5,97.5], axis=0) # p value, CI
        # GET BOOTSTRAPPED DISTR, THEN use summarize
        raise NotImplementedError("single-condition bootstrap not implemented yet")
    elif len(data) == 2:
        out = hierarch.stats.hypothesis_test(matrix, treatment_col=treatment_col, compare=compare, **kwargs)
    else:
        out = hierarch.stats.multi_sample_test(matrix, treatment_col=treatment_col, compare=compare, hypotheses=which, **kwargs)
        if names is not None:
            mapping = {float(i): n for i, n in enumerate(names)}
            out.iloc[:,[0,1]] = out.iloc[:,[0,1]].replace(mapping)

    return out