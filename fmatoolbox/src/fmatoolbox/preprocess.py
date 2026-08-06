''' Preprocessing routines for electrophysiological data '''

import fmatoolbox.analysis
import numpy as np


def corticalDownStates(spikes, n:int=None, silence:float=None, return_fr=None):
    '''
    find cortical down states from population spiking activity

    arguments:
        spikes       (:,) float, cortical spike times
        n            int = 1, number of spikes allowed during a down state
        silence      float = 0.12 s, minimum cortical silence duration
        return_fr    bool = False, return population firing rate (useful to plot results)

    output:
        down         (:,2) float, every row is [start, stop] (s) of a down state
        fr           (:,2) float, every row is [time (s), cortical firing rate (Hz)] (optional)
    '''

    if n is None: n = 1
    n = n + 1
    if silence is None: silence = 0.12

    distances = spikes[n:] - spikes[:-n]
    is_down = distances > silence
    down = spikes[:-n][is_down]
    down = np.stack((down, down+distances[is_down]), axis=1)
    durations = distances[is_down]

    # if two down states intersect, take longest one
    _, is_overlap = fmatoolbox.general.restrict(down[:,1]+1e-10,down,s_ind=True) # downs which overlap with next one
    keep = np.full_like(is_overlap,True)
    remove = np.where(is_overlap)[0] # remove[i] is idx of down to drop
    change = durations[remove] > durations[remove+1] # change remove[i] if next down is smaller
    remove[change] += 1
    keep[remove] = False
    down = down[keep]

    if return_fr:
        fr = fmatoolbox.analysis.istantaneousRate(spikes,bin=0.015,step=4)
        return down, fr

    return down