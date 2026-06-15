import pathlib
import numpy as np
import re
import fmatoolbox.analysis
import fmatoolbox.data
import regions.computation
import regions.loaders


class Regions:
    # Handler for multi-region spiking data, stores session metadata and provides access to computed quantities

    def __init__(self,session,ids=None,phases=None,states=None,events=None,load_spikes=True,reload=False):
        # construct a Regions object
        #
        # arguments:
        #     session        string, path to session .xml file
        #     ids            (:) string = None, regions to load (default is all recorded regions)
        #     phases         (:) string = None, session phases to load from <basename>.cat.evt file
        #     states         (:) string = None, behavioral states to load (they correspond to extensions of files to load)
        #     events         (:) string = None, additional events to load (they correspond to extensions of files to load)
        #     load_spikes    bool = True, load spikes (False allows to access events without costly spike loading)
        #     reload         bool = False, load spikes from original files, bypassing Regions/<basename>_spikes.npz backup

        self.session = pathlib.Path(session).parent
        self.basename = pathlib.Path(session).name

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
        loaded_events = fmatoolbox.data.loadEvents(session,extra=states+events)
        states = [s.rsplit('/',1)[-1] for s in states] # remove everything before '/' in every event
        events = [e.rsplit('/',1)[-1] for e in events]
        # find session phases, if any
        phase_names = [name for name in loaded_events.keys() if name not in states and name not in events]

        if phases:
            indices = [phase_names.index(p) for p in phases if p in phase_names]
            unknown = set(events) - set(phase_names)
            if unknown:
                print(f'Warning: missing events: {unknown}')
            self.phases = {phase_names[i] : loaded_events[phase_names[i]] for i in indices}
        else:
            self.phases = {name : loaded_events[name] for name in phase_names}

        # 2. assign states
        self.states = {}
        for name in states:
            if name not in loaded_events:
                raise ValueError(f'Unable to load {self.basename}.{name}')
            self.states[name] = loaded_events[name]
        # if session phases are available, use them to compute special states 'all' and 'other'
        if phase_names:
            self.states['all'] = np.array([[self.phases[list(self.phases)[0]][0,0],self.phases[list(self.phases)[-1]][-1,-1]]])
            self.states['other'] = self.states['all']
            for name in states:
                self.states['other'] = fmatoolbox.general.subtractIntervals(self.states['other'],self.states[name])

        # 3. assign events
        self.events = {}
        if events:
            for name in events:
                e = loaded_events[name]
                # special reordering for ripples and spindles
                if name in ['ripples','spindles']:
                    e = e[:,[0,2,1]+list(range(3,e.shape[1]))]
                self.events[name] = e

        if ids:
            self.ids = list(dict.fromkeys(ids))
            self.region = {id : {} for id in ids}
        else:
            self.ids = ids
            self.region = {}

        # 4. load spikes and store them per region
        if load_spikes:
            self.region = fmatoolbox.data.loadSpikeTimes(session,output='regions',anat_file=regions.loaders.regionDataPath()/'nonlateral.anat',reload=reload)
            if ids:
                self.region = {r: self.region[r] for r in ids}
            else:
                self.ids = np.array(list(self.region.keys()),dtype=str)

        return
    

    ## validation functions ##

    def _checkIDs(self,regs=None,states=None,fuse=False):
        # validate that regions and states are loaded in self
        #
        # arguments:
        #     regs      (:) string = None, brain regions, default is all loaded regions
        #     states    (:) string = None, behavioral states, default depends on fuse
        #     fuse      bool = False, if True, default states is 'all', else it is all other states
        #
        # output:
        #     regs      (:) string, unique regs (preserving order)
        #     states    (:) string, unique states (preserving order)

        # default: all regions
        if regs is None:
            regs = self.ids
        else:
        # return unique regs
            regs = np.array(regs).flatten()
            if not np.isin(regs,self.ids).all():
                raise(ValueError(f'Unrecognized region'))
            _, idx = np.unique(regs,return_index=True)
            regs = regs[np.sort(idx)]

        # default:
        if states is None:
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
                raise(ValueError(f'Unrecognized state'))
            _, idx = np.unique(states,return_index=True)
            states = states[np.sort(idx)]

        return regs, states
    

    ## getters with minimal processing ##

    def eventIntervals(self,events=None):
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
        #
        # output:
        #     intervals    (:,2) double, each row is a [start, stop] interval (s)

        def match(events,patterns):
            matches = []
            for pattern in patterns:
                m = re.fullmatch(r"(.*)#(\d+)",pattern) # look for #
                if m:
                    # take match indexed by 'idx': digit after # (or none)
                    pat, idx = m.groups()
                    idx = int(idx)
                    match = [e for e in events if re.fullmatch(pat,e)]
                    if 0 <= idx < len(match):
                        matches.append(match[idx])
                else:
                    # usual regexp
                    matches += [e for e in events if re.fullmatch(pattern,e)]
            return matches

        # default output
        if events is None:
            intervals = np.concatenate(list(self.phases.values()))

        # list of events
        else:

            # promote single string to 2d array
            if isinstance(events,str):
                events = np.array(events,ndmin=2)

            # events is a list of lists of event names
            intervals = []
            for ev in events:
                # 1. union of all intervals in ev
                ev = np.asarray(ev)
                if ev.ndim == 0:
                    raise ValueError("'events' must be like a list of lists of strings")
                interv = [self.phases[e][:,:2] for e in match(self.phases,ev)]
                [interv.append(self.states[e][:,:2]) for e in match(self.states,ev)]
                [interv.append(self.events[e][:,:2]) for e in match(self.events,ev)]
                if len(interv) == 0:
                    raise ValueError(f"None of the following was found: {ev}")
                intervals.append(fmatoolbox.general.consolidateIntervals(np.concatenate(interv)))
            # 2. intersection across different evs
            intervals = fmatoolbox.general.intersectIntervals(intervals)
                
        return intervals
    

    def eventInfo(self,event):
        # get event information matrix
        #
        # arguments:
        #     event    string, event about which to get information
        #
        # output:
        #     info     (:,:) float, event information, each row corresponds to an instance of the event

        if event not in self.events:
            raise ValueError(f'Unable to find event {event}')
        
        info = self.events[event]

        return info
    

    def units(self,regs=None,e_groups=None):
        # get pooled list of units for regions
        #
        # arguments:
        #     regs        (:) string, units of all these regions are returned as an array
        #     e_groups    (:) int, units of all these electrode groups (starting at 1) are returned as an array
        #
        # output:
        #     units       (:) int, sorted by value

        temp_flag = regs is None and e_groups is not None
        regs, _ = self._checkIDs(regs)
        if temp_flag:
            regs = [None]
        if isinstance(e_groups,int):
            e_groups = [e_groups]

        units = []
        for r in self.ids:
            if r in regs:
                units.extend(list(self.region[r]['e_group'].values()))
            elif e_groups is not None:
                [units.append(u) for g, u in self.region[r]['e_group'].items() if g in e_groups]

        return np.sort(np.concatenate(units)) if len(units) else units


    def spikes(self,regs=None,state=None,when=None,shift=False):
        # get pooled spikes for regions
        #
        # arguments:
        #     regs      (:) string, spikes of all these regions are returned as a time-sorted array
        #     state     string = None, behavioral to restrict spikes to
        #     when      DESCRIBE, same input as eventIntervals
        #     shift     bool = False, shift epochs together in time after filtering by state
        #
        # output:
        #     spikes    (:) float, each row is [spike time (s), unit id]

        regs, state = self._checkIDs(regs,state,fuse=True)

        spikes = np.concatenate([self.region[r]['spikes'] for r in regs])
        spikes = spikes[spikes[:,0].argsort()] # sort by time

        if np.any(state != 'all'):
            spikes = fmatoolbox.general.restrict(spikes,self.eventIntervals([state]),shift=shift)
        if when is not None:
            spikes = fmatoolbox.general.restrict(spikes,self.eventIntervals(when),shift=shift)

        return spikes
    

    ## functions to compute quantities ##

    def firingRate(self,regs=None,states=None,when=None,shift=False,window=0.05,step=1,smooth=None,norm=False):
        # get region firing rate
        #
        # arguments:
        #     regs      (n) string = None, brain regions, default is all loaded regions
        #     states    (:) string = None, behavioral states, default is all
        #     when      DESCRIBE, same input as eventIntervals
        #     shift     bool = False, shift epochs together in time after filtering by state
        #     window    float = 0.05, window size to count spikes
        #     step      int = 1, firing rate is computed in windows of length 'binSize' and overlap 'binSize' / 'step',
        #               default is no overlap
        #     smooth    float = None, gaussian kernel std for smoothing over time
        #     norm      bool = False, normalize by neuron number per region
        #
        # output:
        #     rate      (:,n+1) float, every row is [time stamp, firing rates for n regions]

        regs, states = self._checkIDs(regs,states,fuse=True)

        # operate per session phase
        phase_intervals = fmatoolbox.general.consolidateIntervals(self.eventIntervals(),epsilon=0.00001)
        firing_rate = []
        time = []
        for interval in phase_intervals:
            fr_interv = []
            for r in regs:
                fr = fmatoolbox.analysis.firingRate(self.spikes(r)[:,0],interval[0],interval[1],window,step,smooth)
                fr_interv.append(fr[:,1])
            firing_rate.append(np.stack(fr_interv,1))
            time.append(fr[:,0])
        firing_rate = np.concatenate((np.concatenate(time).reshape((-1,1)),np.concatenate(firing_rate)),1)

        # filter by state
        if np.any(states != 'all'):
            firing_rate = fmatoolbox.general.restrict(firing_rate,self.eventIntervals([states]),shift=shift)
        if when is not None:
            firing_rate = fmatoolbox.general.restrict(firing_rate,self.eventIntervals(when),shift=shift)

        # normalize
        if norm:
            firing_rate[:,1:] /= [len(self.units(r)) for r in regs]

        return firing_rate
    

    def unitFiringRate(self,regs=None,states=None,when=None,shift=False,window=0.05,step=1,smooth=None):
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

        regs, states = self._checkIDs(regs,states,fuse=True)

        # operate per session phase
        phase_intervals = fmatoolbox.general.consolidateIntervals(self.eventIntervals(),epsilon=0.00001)
        n_times = np.concatenate(([0],np.cumsum(np.ceil(np.diff(phase_intervals,1)*step/window)).astype(int)))
        n_units = np.cumsum(np.concatenate(([1],[self.units(r).size for r in regs])))
        firing_rate = np.zeros((n_times[-1],n_units[-1]))
        for i, interval in enumerate(phase_intervals):
            for j, r in enumerate(regs):
                fr = fmatoolbox.analysis.firingRate(self.spikes(r),interval[0],interval[1],window,step,smooth)
                firing_rate[n_times[i]:n_times[i+1],n_units[j]:n_units[j+1]] = fr[:,1:]
            firing_rate[n_times[i]:n_times[i+1],0] = fr[:,0]

        # filter by state
        if np.any(states != 'all'):
            firing_rate = fmatoolbox.general.restrict(firing_rate,self.eventIntervals([states]),shift=shift)
        if when is not None:
            firing_rate = fmatoolbox.general.restrict(firing_rate,self.eventIntervals(when),shift=shift)

        return firing_rate
    

    def avalanches(self,regs=None,states=None,when=None,shift=False,thresh=30,window=0.05,step=1,smooth=None):
        # compute avalanches per region from population firing rate
        #
        # arguments:
        #     regs      (:) str = None, brain regions, default is all loaded regions
        #     states    (:) str = None, behavioral states, default is all
        #     when      DESCRIBE, same input as eventIntervals
        #     shift     bool = False, shift epochs together in time after filtering by state
        #     thresh    float = 30, percentile to use as threshold, must be in [0,100]
        #     window    float = 0.05 s, window size to count spikes
        #     step      float = 1, firing rate is computed in windows of length 'binSize' and overlap 'binSize' / 'step',
        #               default is no overlap
        #     smooth    float = None, gaussian kernel std for smoothing over time
        #
        # output:
        #     sizes        (n) float, avalanche sizes
        #     intervals    (n,2) float, each row is an avalanche's [start, stop] interval (s)
        #     size_t       (m) float, size over time, in which every avalanche is separated by a 0

        regs, states = self._checkIDs(regs,states,fuse=True)

        fr = self.firingRate(regs=regs,states=states,when=when,shift=shift,window=window,step=step,smooth=smooth)
        size = {}
        intervals = {}
        size_t = {}
        for i, r in enumerate(regs):
            size[r], intervals[r], size_t[r] = fmatoolbox.analysis.avalanchesFromProfile(fr[:,i+1],thresh,time_step=fr[1,0]-fr[0,0],t0=fr[0,0])

        return size, intervals, size_t