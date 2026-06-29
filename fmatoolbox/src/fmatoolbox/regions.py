''' Handler for multi-region spiking data, stores session metadata and provides access to computed quantities '''

import pathlib
import numpy as np
import re
import fmatoolbox.analysis
import fmatoolbox.data
import warnings
from collections.abc import Iterable


def _regionDataPath():
    # get path to fmatoolbox/data/regions/ folder

    # find scr/ directory, data/ must be at same level
    file_path = pathlib.Path(__file__).resolve()
    parts = file_path.parts
    if 'src' not in parts:
        raise ValueError(f"src/ not found in path")
    idx = parts.index('src')

    return pathlib.Path(*parts[:idx]) / 'data/regions'


class regions:
    # Handler for multi-region spiking data, stores session metadata and provides access to computed quantities

    def __init__(self,session,ids=None,phases=None,states=None,events=None,load_spikes=True,reload=False,anat_file=None):
        # construct a regions object
        #
        # arguments:
        #     session        string, path to session .xml file
        #     ids            (:) string = None, regions to load (default is all recorded regions)
        #     phases         (:) string = None, session phases to load from <basename>.cat.evt file
        #     states         (:) string = None, behavioral states to load (they correspond to extensions of files to load)
        #     events         (:) string = None, additional events to load (they correspond to extensions of files to load)
        #     load_spikes    bool = True, load spikes (False allows to access events without costly spike loading)
        #     reload         bool = False, load spikes from original files, bypassing Regions/<basename>_spikes.npz backup
        #     anat_file      string = None, DESCRIBE

        self.session = pathlib.Path(session).parent
        self.basename = pathlib.Path(session).stem
        self.rat = self.basename[3:6]

        # 1. load events
        self.all_events = phases is None
        if states is None:
            states = []
        elif isinstance(states,str):
            states = [states]
        if events is None:
            events = []
        elif isinstance(events,str):
            events = [events]
        if isinstance(phases,str):
            phases = [phases]
        loaded_events, phase_names = fmatoolbox.data.loadEvents(session,extra=states+events)
        states = [s.rsplit('/',1)[-1] for s in states] # remove everything before '/' in every event
        events = [e.rsplit('/',1)[-1] for e in events]

        # 2. assign session epochs
        matches = self._matchEvents(phase_names,phases) if phases else phase_names
        self.phases = {m: np.stack((loaded_events[m]['beginning'],loaded_events[m]['end']),axis=1) for m in matches}

        # 3. assign states
        self.states = {s: np.stack((loaded_events[s]['col0'],loaded_events[s]['col1']),axis=1) for s in states}
        # if session phases are available, use them to compute special states 'all' and 'other' SHOULD ALSO RESTRICT states AND events TO phases
        if phase_names:
            self.states['all'] = np.array([[self.phases[list(self.phases)[0]][0,0],self.phases[list(self.phases)[-1]][-1,-1]]])
            self.states['other'] = self.states['all']
            for name in states:
                self.states['other'] = fmatoolbox.general.subtractIntervals(self.states['other'],self.states[name])

        # 3. assign events
        self.events = {}
        for name in loaded_events:
            if name not in phase_names and name not in states:
                e = loaded_events[name]

                attributes = list(e.keys())
                if 'timestamps' in attributes:
                    e['intervals'] = e['timestamps']
                    e.pop('timestamps')
                else:
                    if name in ['ripples','spindles']:
                        # special reordering for ripples and spindles
                        end = np.intersect1d(attributes,['col2','end','stop'])[0]
                        if 'col1' in attributes:
                            e['peak'] = e['col1']
                            e.pop('col1')
                    else:
                        end = np.intersect1d(attributes,['col1','end','stop'])[0]
                    start = np.intersect1d(attributes,['col0','beginning','start'])[0]
                    e['intervals'] = np.stack((e[start],e[end]),axis=1)
                    e.pop(start)
                    e.pop(end)
                self.events[name] = e

        if ids:
            self.ids = list(dict.fromkeys(ids))
            self.region = {id : {} for id in ids}
        else:
            self.ids = ids
            self.region = {}

        # 4. load spikes and store them per region
        if load_spikes:
            if anat_file is None:
                anat_file = _regionDataPath() / 'nonlateral.anat'
            else:
                anat_file = pathlib.Path(anat_file)
                if not anat_file.exists():
                    anat_file = _regionDataPath() / anat_file
            self.region = fmatoolbox.data.loadSpikeTimes(session,output='regions',anat_file=anat_file,reload=reload)
            if ids:
                self.region = {r: self.region[r] for r in ids}
            else:
                self.ids = np.array(list(self.region.keys()),dtype=str)
            if not self.all_events:
                # restrict spikes to session phases
                for id in self.ids:
                    self.region[id]['spikes'] = fmatoolbox.general.restrict(self.region[id]['spikes'],self.eventIntervals())

        return
    

    ## validation functions ##

    def _checkIDs(self,regs=None,e_groups=None,states=None,exclusive=True,fuse=False):
        # validate that regions, electrode groups, and states exist in Regions object
        #
        # arguments:
        #     regs         (:) string = None, brain regions, see 'exclusive' for default value
        #     e_groups     (:) int = None, electrode groups, see 'exclusive' for default value
        #     states       (:) string = None, behavioral states, see 'fuse' for default value
        #     exclusive    bool = True, make default 'regs' (or 'e_groups') [None] when 'e_groups' (or 'regs') is provided,
        #                  otherwise they are all regions and all electrode groups, respectively
        #     fuse         bool = False, if True, default states is "all", else it is all other states
        #
        # output:
        #     regs        (:) string, unique regs (preserving order)
        #     e_groups    (:) int, unique electrode groups (preserving order)
        #     states      (:) string, unique states (preserving order)

        regs = np.asarray(regs)
        e_groups = np.asarray(e_groups)
        states = np.asarray(states)

        # default: all regions
        if np.all(regs == None):
            if exclusive and np.any(e_groups != None):
                regs = np.array([None])
            else:
                regs = self.ids
        else:
        # return unique regs
            regs = np.array(regs).flatten()
            if not np.isin(regs,self.ids).all():
                raise ValueError('unrecognized region')
            _, idx = np.unique(regs,return_index=True)
            regs = regs[np.sort(idx)]

        # default: all electrode groups
        all_e_groups = np.concatenate([list(self.region[r]['e_group'].keys()) for r in self.ids])
        if np.all(e_groups == None):
            if exclusive and np.any(regs != None):
                e_groups = np.array([None])
            else:
                e_groups = all_e_groups
        else:
            e_groups = np.array(e_groups).flatten()
            if not np.isin(e_groups,all_e_groups).all():
                raise ValueError('unrecognized electrode group')
            _, idx = np.unique(e_groups,return_index=True)
            e_groups = e_groups[np.sort(idx)]

        # default:
        if np.all(states == None):
            # 'all' if fuse
            if fuse or len(self.states.keys()) == 2:
                states = np.array(['all'])
            # all states otherwise
            else:
                states = np.array(list(self.states))
                states = states[states != 'all']
        # return unique states
        else:
            states = np.array(states).flatten()
            if not np.isin(states,list(self.states.keys())).all():
                raise ValueError('unrecognized state')
            _, idx = np.unique(states,return_index=True)
            states = states[np.sort(idx)]

        return regs, e_groups, states


    def _matchEvents(self,events,patterns):

        matches = []
        for pattern in patterns:
            m = re.fullmatch(r"(.*)#(\d+(?:,\d+)*)", pattern) # look for #
            if m:
                # take match indexed by 'idx': digits after '#' (or none)
                pat, idx = m.groups()
                idx = [int(i) for i in idx.split(",")]
                match = [e for e in events if re.fullmatch(pat,e)]
                for i in idx:
                    if 0 <= i < len(match):
                        matches.append(match[i])
            else:
                # usual regexp
                matches += [e for e in events if re.fullmatch(pattern,e)]
        return matches

    ## getters with minimal processing ##

    def eventIntervals(self,events=None,epsilon=0):
        # get [start, stop] intervals (s) for a union and/or intersection of events
        #
        # arguments:
        #     events       (n) list of (:) string, each element is a list of events; to compute intervals:
        #                    1. intervals corresponding to names from each list inside 'events' are united, yielding n interval sets
        #                    2. output is intersection between this n sets
        #                  e.g., events = [['rem','sws'],['sleep1']]
        #                    1. 'rem' and 'sws' intervals are united (a), 'sleep1' is unchanged (b)
        #                    2. intersection between (a) and (b) is output
        #                  note: event names are interpreted as regular expressions and searched with re.fullmatch;
        #                    if a name ends in # followed by digits, they are interpreted as the index of the match to keep
        #                  e.g., 'sleep.*' matches all sleep events, 'sleep.*2' matches the third sleep event (if present)
        #     epsilon      float = 0, intervals with bounds closer than 'epsilon' are consolidated
        #
        # output:
        #     intervals    (:,2) double, each row is a [start, stop] interval (s)

        # 1. default output
        if events is None:
            return np.concatenate(list(self.phases.values()))

        # 2. numeric intervals
        try:
            intervals = np.array(events,ndmin=2,dtype=float)
            return intervals
        except ValueError:
            pass

        # 3. list of lists of event names

        # promote single string to 2d array
        if isinstance(events,str):
            events = np.array(events,ndmin=2)

        intervals = []
        for ev in events:
            # 1. union of all intervals in ev
            ev = np.asarray(ev)
            if ev.ndim == 0:
                raise ValueError("'events' must be like a list of lists of strings")
            interv = [self.phases[e][:,:2] for e in self._matchEvents(self.phases,ev)]
            [interv.append(self.states[e][:,:2]) for e in self._matchEvents(self.states,ev)]
            [interv.append(self.events[e]['intervals'][:,:2]) for e in self._matchEvents(self.events,ev)]
            if len(interv) == 0:
                raise ValueError(f"None of the following was found: {ev}")
            intervals.append(fmatoolbox.general.consolidateIntervals(np.concatenate(interv),epsilon=epsilon))
        # 2. intersection across different evs
        intervals = fmatoolbox.general.intersectIntervals(intervals)
                
        return intervals
    

    def eventInfo(self,event,attribute):
        # get event information, corresponding to a field of the event dictionary
        #
        # arguments:
        #     event    string, event about which to get information
        #
        # output:
        #     info     (:,:) float, event information, each row corresponds to an instance of the event

        if event not in self.events:
            raise ValueError(f'Unable to find event {event}')

        if attribute not in self.events[event]:
            if attribute == 'start':
                return self.events[event]['intervals'][:,0]
            elif attribute == 'stop':
                return self.events[event]['intervals'][:,1]
            else:
                raise ValueError(f'{event} has no attribute {attribute}, valid attributes are: ' + ', '.join(self.events[event].keys()))

        return self.events[event][attribute]


    def electrodes(self,regs=None):
        # get pooled list of electrode groups for regions
        #
        # arguments:
        #     regs        (:) string = None, electrode of these regions are returned as an array, default is all regions
        #
        # output:
        #     electrodes       (:) int, sorted by value

        regs, _, _ = self._checkIDs(regs=regs)

        return np.concatenate([list(self.region[r]['e_group'].keys()) for r in regs])


    def units(self,regs=None,e_groups=None):
        # get pooled list of units for regions
        #
        # arguments:
        #     regs        (:) string = None, units of all these regions are returned as an array, default is all regions
        #     e_groups    (:) int = None, units of all these electrode groups (starting at 1) are returned as an array
        #
        # output:
        #     units       (:) int, sorted by value

        regs, e_groups, _ = self._checkIDs(regs=regs,e_groups=e_groups)

        units = []
        for r in self.ids:
            if r in regs:
                units.extend(list(self.region[r]['e_group'].values()))
            elif np.any(e_groups != [None]):
                [units.append(u) for g, u in self.region[r]['e_group'].items() if g in e_groups]

        return np.sort(np.concatenate(units)) if len(units) else units


    def spikes(self,regs=None,e_groups=None,state=None,when=None,shift=False):
        # get pooled spikes for regions
        #
        # arguments:
        #     regs        (:) string = None, spikes of all these regions are returned as a time-sorted array
        #     e_groups    (:) int = None, units of all these electrode groups (starting at 1) are returned as a time-sorted array
        #     state       string = None, behavioral to restrict spikes to
        #     when        DESCRIBE, same input as eventIntervals
        #     shift       bool = False, shift epochs together in time after filtering by state
        #
        # output:
        #     spikes    (:) float, each row is [spike time (s), unit id]

        if state is not None and when is not None:
            raise ValueError("'state' and 'when' cannot be specified at the same time")

        regs, e_groups, state = self._checkIDs(regs=regs,e_groups=e_groups,states=state,fuse=True)

        spikes = []
        e_group_units = self.units(e_groups=e_groups)
        for r in self.ids:
            if r in regs:
                spikes.append(self.region[r]['spikes'])
            elif np.any(e_groups != [None]):
                s = self.region[r]['spikes']
                spikes.append(s[np.isin(s[:,1],e_group_units),:])

        spikes = np.concatenate(spikes)
        spikes = spikes[spikes[:,0].argsort()] # sort by time

        if np.any(state != 'all'):
            spikes = fmatoolbox.general.restrict(spikes,self.eventIntervals([state]),shift=shift)
        if when is not None:
            try:
                # 1. 'when' is a list of time intervals
                spikes = fmatoolbox.general.restrict(spikes,when,shift=shift)
            except:
                # 2. 'when' contains event names
                spikes = fmatoolbox.general.restrict(spikes,self.eventIntervals(when),shift=shift)

        return spikes
    

    ## functions to compute quantities ##

    def firingRate(self, regs:Iterable[str]=None, e_groups:Iterable[int]=None, states:Iterable[str]=None, when=None, shift=False,
                   window=None, step=None, smooth=None, norm=False):
        # get region firing rate
        #
        # arguments:
        #     regs        (n) string = None, brain regions to compute firing rate of, default is all regions
        #     e_groups    (m) int = None, electrode groups (starting at 1) to compute firing rate of, default is none
        #     states      (:) string = None, behavioral states, default is all
        #     when        DESCRIBE, same input as eventIntervals
        #     shift       bool = False, shift epochs together in time after filtering by state
        #     window      float = 0.05, window size to count spikes
        #     step        int = 1, firing rate is computed in windows of length 'binSize' and overlap 'binSize' / 'step',
        #                 default is no overlap
        #     smooth      float = None, gaussian kernel std for smoothing over time
        #     norm        bool = False, normalize by neuron number per region
        #
        # output:
        #     rate        (:,n+m+1) float, every row is [time stamp, firing rates for n regions, firing rates for m electrodes],
        #                 input order of regions and electrodes is preserved

        if states is not None and when is not None:
            raise ValueError("'states' and 'when' cannot be specified at the same time")
        if window is None: window = 0.05
        if step is None: step = 1

        regs, e_groups, states = self._checkIDs(regs=regs,e_groups=e_groups,states=states,fuse=True)

        # find big holes in 'when' intervals, to speed up computation, NOTE: might not be speeding it up after all...
        when = self.eventIntervals(when)
        holes = when[1:,0] - when[:-1,1] > 5000 # s
        partition_idx = np.insert(np.cumsum(holes),0,0) # partition_idx[i] indexes partition to which when[i] belongs
        partitions = np.unique(partition_idx)
        do_restrict = shift or len(partitions) != len(partition_idx)

        # operate per partition
        firing_rate = []
        time = []
        for p in partitions:
            intervals = when[partition_idx == p]
            fr_interv = []

            if np.any(regs != None):
                for r in regs:
                    fr = fmatoolbox.analysis.firingRate(self.spikes(regs=r)[:,0],np.floor(intervals[0,0]),intervals[-1,1],window,step,smooth)
                    fr_interv.append(fr[:,1])

            if np.any(e_groups != None):
                for e in e_groups:
                    fr = fmatoolbox.analysis.firingRate(self.spikes(e_groups=e)[:,0],np.floor(intervals[0,0]),intervals[-1,1],window,step,smooth)
                    fr_interv.append(fr[:,1])

            firing_rate.append(np.stack(fr_interv,1))
            time.append(fr[:,0])
        firing_rate = np.concatenate((np.concatenate(time).reshape((-1,1)),np.concatenate(firing_rate)),1)

        if do_restrict:
            firing_rate = fmatoolbox.general.restrict(firing_rate,when,shift=shift)

        # filter by state
        if np.any(states != 'all'):
            firing_rate = fmatoolbox.general.restrict(firing_rate,self.eventIntervals([states]),shift=shift)
            warnings.warn("option 'states' is deprecated, use 'when' instead")

        # normalize
        if norm:
            with np.errstate(divide='ignore',invalid='ignore'):
                firing_rate[:,1:] /= [len(self.units(regs=r)) for r in regs if r is not None]+[len(self.units(e_groups=g)) for g in e_groups if g is not None]

        return firing_rate
    

    def unitFiringRate(self,regs=None,states=None,when=None,shift=False,window=None,step=None,smooth=None):
        # get units' firing rate
        #
        # arguments:
        #     regs      (:) str = None, brain regions, default is all loaded regions
        #     states    (:) str = None, behavioral states, default is all
        #     when      DESCRIBE, same input as eventIntervals
        #     shift     bool = False, shift epochs together in time after filtering by state
        #     window    float = 0.05 s, window size to count spikes
        #     step      int = 1, firing rate is computed in windows of length 'binSize' and overlap 'binSize' / 'step',
        #               default is no overlap
        #     smooth    float = None, gaussian kernel std for smoothing over time
        #
        # output:
        #     rate      (:,n+1) float, every row is [time stamp, firing rates for n units]

        if states is not None and when is not None:
            raise ValueError("'states' and 'when' cannot be specified at the same time")
        if window is None: window = 0.05
        if step is None: step = 1

        regs, _, states = self._checkIDs(regs=regs,states=states,fuse=True)

        # operate per session phase
        phase_intervals = fmatoolbox.general.consolidateIntervals(self.eventIntervals(),epsilon=0.00001)
        n_times = np.concatenate(([0],np.cumsum(np.ceil(np.diff(phase_intervals,1)*step/window)).astype(int)))
        n_units = np.cumsum(np.concatenate(([1],[self.units(r).size for r in regs])))
        firing_rate = np.zeros((n_times[-1],n_units[-1]))
        for i, interval in enumerate(phase_intervals):
            for j, r in enumerate(regs):
                fr = fmatoolbox.analysis.firingRate(self.spikes(r),interval[0],interval[1],window,step,smooth,u_range=self.units(r))
                firing_rate[n_times[i]:n_times[i+1],n_units[j]:n_units[j+1]] = fr[:,1:]
            firing_rate[n_times[i]:n_times[i+1],0] = fr[:,0]

        # filter by state
        if np.any(states != 'all'):
            firing_rate = fmatoolbox.general.restrict(firing_rate,self.eventIntervals([states]),shift=shift)
        if when is not None:
            try:
                # 1. 'when' is a list of time intervals
                firing_rate = fmatoolbox.general.restrict(firing_rate,when,shift=shift)
            except:
                # 2. 'when' contains event names
                firing_rate = fmatoolbox.general.restrict(firing_rate,self.eventIntervals(when),shift=shift)

        return firing_rate
    

    def avalanches(self,regs=None,states=None,when=None,shift=False,thresh=30,window=0.05,step=1,smooth=None,return_fr=False,norm=False):
        # compute avalanches per region from population firing rate
        #
        # arguments:
        #     regs         (r) str = None, brain regions, default is all loaded regions
        #     states       (:) str = None, behavioral states, default is all
        #     when         DESCRIBE, same input as eventIntervals
        #     shift        bool = False, shift epochs together in time after filtering by state
        #     thresh       float = 30, percentile to use as threshold, must be in [0,100]
        #     window       float = 0.05 s, window size to count spikes
        #     step         float = 1, firing rate is computed in windows of length 'binSize' and overlap 'binSize' / 'step',
        #                  default is no overlap
        #     smooth       float = None, gaussian kernel std for smoothing over time
        #     norm         bool = False, normalize by neuron number per region
        #
        # output:
        #     sizes        (n) float, avalanche sizes
        #     intervals    (n,2) float, each row is an avalanche's [start, stop] interval (s)
        #     size_t       (m) float, size over time, in which every avalanche is separated by a 0
        #     fr           (:,r+1) float, every row is [time stamp, firing rates for r regions], optional

        none_state = states is None
        regs, _, states = self._checkIDs(regs=regs,states=states,fuse=True)
        if none_state:
            states = None

        fr = self.firingRate(regs=regs,states=states,when=when,shift=shift,window=window,step=step,smooth=smooth,norm=norm)
        size = {}
        intervals = {}
        size_t = {}
        for i, r in enumerate(regs):
            size[r], intervals[r], size_t[r] = fmatoolbox.analysis.avalanchesFromProfile(fr[:,i+1],thresh,time_step=fr[1,0]-fr[0,0],t0=fr[0,0])

        if return_fr:
            return size, intervals, size_t, fr
        return size, intervals, size_t