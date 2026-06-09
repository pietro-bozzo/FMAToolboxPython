import numpy as np

# DEPRECATED
def avalanchesFromProfile(x,threshold,time_step,t0=0):
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
    threshold = np.percentile(x,threshold)
    x = x - threshold
    x[x<0] = 0

    is_ok = np.concatenate(([True],(x[1:] != 0) | (x[:-1] != 0))) # is_ok[i] = 0 if i-th element is repeated zero

    # sizes
    size_t = x[is_ok] * time_step # remove repeated zeros, obtaining size per bin: size over time
    sizes = np.bincount(np.cumsum(size_t==0) - (x[0] == 0), weights=size_t)
    # remove last zero
    if sizes[-1] == 0:
        sizes = sizes[:-1]
    
    #  start and stop times
    start = np.where(np.concatenate(([x[0] != 0], (x[1:] != 0) & (x[:-1] == 0))))[0]
    stop = np.where(np.concatenate(((x[1:] == 0) & (x[:-1] != 0), [x[-1] != 0])))[0] + 1
    intervals = np.stack((start,stop),1) * time_step + t0 - time_step/2

    return sizes, intervals, size_t