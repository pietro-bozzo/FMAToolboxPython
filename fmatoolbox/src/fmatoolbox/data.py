''' Session data handling functions for FMAToolbox '''
from IPython.core.interactiveshell import is_integer_string
from typing import Any, Callable
import pathlib
import scipy.io
import numpy as np
import ast
import datetime
import traceback
import re
import fmatoolbox.exceptions
import collections
import concurrent.futures


def loadSpikeTimes(session:str, output:str='dict', anat_file:str=None, return_elec:bool=False, return_loc:bool=False, reload:bool=False):
    # load spikes from a session
    #
    # arguments:
    #     session         string, path to session .xml file, spike files must be in session folder
    #     output          string = None, determines output type, can be:
    #                       'dict', dictionary of spike times per unit
    #                       'compact', (:,2) array of [timestamps, unit ids]
    #                       'full', (:,2) array of [timestamps, electrode groups, clusters]
    #                       'regions', dict of {region id: dict}, entries are {'spikes': region spikes, 'units': region units}
    #     anat_file        str = None, DESCRIBE
    #     return_elec      bool = False, if true, return cluster_loc
    #     return_loc       bool = False, if true, return cluster_loc
    #     reload           bool = False, load original files, bypassing Regions/<basename>_spikes.npz backup (for 'output' = 'regions')
    #
    # output:
    #     spikes          (see 'output')
    #     electrodes      (n) float, optional, electrode id per unit
    #     cluster_loc     (n) float, optional, index of max spike-amplitude cluster per unit

    file_root = pathlib.Path(session).with_suffix('')
    cell_metrics_file = file_root.with_suffix('.cell_metrics.cellinfo.mat')
    if cell_metrics_file.exists():
        if return_elec or return_loc:
            spikes, electrode_id, cluster_loc = loadCellMetricsFile(session,output=output,anat_file=anat_file,return_extra=True,reload=reload)
        else:
            spikes = loadCellMetricsFile(session,output=output,anat_file=anat_file,reload=reload)
    else:
        if return_elec or return_loc:
            raise ValueError("'return_elec' and 'return_loc' are not implemented for .clu files")
        spikes = loadCluFiles(session,output=output,anat_file=anat_file,reload=reload)

    if not return_elec and not return_loc:
        return spikes
    out = (spikes, electrode_id, cluster_loc) # prepare tuple to return requested outputs
    return out[:2+return_loc:2-return_elec]


def loadCellMetricsFile(session:str, output:str='dict', anat_file:str=None, return_extra:bool=False, reload:bool=False):

    file_root = pathlib.Path(session).with_suffix('')

    # load .cell_metrics.cellinfo.mat file only if necessary
    if output != 'regions' or reload or return_extra:
        cell_metrics_file = file_root.with_suffix('.cell_metrics.cellinfo.mat')
        if not cell_metrics_file.exists():
            raise FileNotFoundError(f'{cell_metrics_file} not found')

        data = scipy.io.loadmat(cell_metrics_file,simplify_cells=True)['cell_metrics']
        unit_id = data['UID'] - 1
        electrode_id = data['electrodeGroup'] # starts at 1
        # starts from 0 CHECK WITH CLUSTER LOC, there's also maxWaveformChannelOrder, there's also Putative cell type!!
        cluster_loc = data['maxWaveformCh']
        spikes = data['spikes']['times']

    if output == 'dict':
        spikes = dict(zip(unit_id,spikes))

    elif output == 'compact':
        ids = np.repeat(unit_id, [len(s) for s in spikes])
        spikes = np.stack((np.concatenate(spikes),ids), axis=1)
        spikes = spikes[spikes[:, 0].argsort()]

    elif output == 'full':
        electrodes = np.repeat(electrode_id, [len(s) for s in spikes])
        cluster_id = data['cluID']  # min is 2, as 0 and 1 are excluded clusters?
        clusters = np.repeat(cluster_id, [len(s) for s in spikes])
        spikes = np.stack((np.concatenate(spikes), electrodes, clusters), axis=1)
        spikes = spikes[spikes[:, 0].argsort()]

    elif output == 'regions':
        # try loading
        spike_file = file_root.parent / 'Regions' / (file_root.stem + '_spikes.npz')
        if not reload and spike_file.exists():
            spikes = loadRegionSpikes(spike_file)
        else:
            spikes_dict = dict(zip(unit_id,spikes))
            if anat_file is None:
                raise ValueError("'anat_file' must be provided when 'output' = 'regions'")
            anat = loadAnatomyFile(anat_file)
            anat = anat[anat['rat'] == int(file_root.stem[3:6])]  # keep rat of interest, deduced from file name
            ids = np.unique(anat['region'])
            spikes = {str(id): {} for id in ids}
            for id in ids:
                spikes[id]['e_group'] = {g: [] for g in np.unique(anat[anat['region'] == id]['electrode'])}
                s = []
                for electrode in spikes[id]['e_group']:
                    electrode_units = unit_id[electrode_id == electrode]
                    spikes[id]['e_group'][electrode] = electrode_units
                    for unit in electrode_units:
                        s.append(np.array([spikes_dict[unit], [unit] * spikes_dict[unit].size]).T)
                # assing spikes, sorted by time
                if len(s):
                    s = np.concatenate(s)
                    spikes[id]['spikes'] = s[s[:,0].argsort()]
                else:
                    spikes[id]['spikes'] = np.array(s,ndmin=2)
            # save
            saveRegionSpikes(spike_file,spikes)

    else:
        raise ValueError("'output' must be 'dict', 'compact', 'full', or 'regions'")

    if return_extra:
        return spikes, electrode_id, cluster_loc
    return spikes


def loadCluFiles(session:str, rate:float=20000, output:str='dict', anat_file:str=None, reload:bool=False):

    if output not in ['dict','compact','full','regions']:
        raise ValueError("'output' must be 'dict', 'compact', 'full', or 'regions'")

    # load .res and .clu files
    froot = pathlib.Path(session).with_suffix('')
    res_files = {int(p.name[len(froot.name)+5:]): p for p in froot.parent.iterdir()
                 if p.is_file() and p.name.startswith(froot.name + '.res.') and p.name[len(froot.name)+5:].isdigit()}
    clu_files = {int(p.name[len(froot.name)+5:]): p for p in froot.parent.iterdir()
                 if p.is_file() and p.name.startswith(froot.name + '.clu.') and p.name[len(froot.name)+5:].isdigit()}
    res = {r: np.loadtxt(res_files[r],comments='%',delimiter=',') for r in res_files}
    clu = {c: np.loadtxt(clu_files[c],comments='%',delimiter=',') for c in clu_files}

    # create valid spike groups (i.e., electrodes where .res and .clu files have compatible length)
    groups = np.sort(list(res.keys()))
    clu_spikes = {}
    for group in groups:
        if group in res and group in clu and len(res[group]) == len(clu[group])-1:
            clu_spikes[group] = np.array([res[group], np.full(len(res[group]),group), clu[group][1:]]).T
            keep = (clu_spikes[group][:,-1] != 0) & (clu_spikes[group][:,-1] != 1) # remove artifacts and MUA
            clu_spikes[group] = clu_spikes[group][keep,:]
            clu_spikes[group][:,0] /= rate
    if not clu_spikes:
        raise FileNotFoundError('no valid .res or .clu file found')

    if output == 'dict':
        # loop through groups, find all units, add to dict
        spikes = {}
        id = 0
        for group in clu_spikes:
            units, idx = np.unique(clu_spikes[group][:,1:],axis=0,return_inverse=True)
            for u in range(len(units)):
                spikes[id] = clu_spikes[group][idx==u,0]
                id += 1

    elif output == 'compact':
        # concatenate all values in spikes, use unique for ids
        spikes = np.concatenate(tuple(clu_spikes.values()),axis=0)
        units, idx = np.unique(spikes[:,1:],axis=0,return_inverse=True)
        spikes = np.stack((spikes[:,0],idx),axis=1)
        spikes = spikes[spikes[:,0].argsort()]

    elif output == 'full':
        spikes = np.concatenate(tuple(clu_spikes.values()),axis=0)
        spikes = spikes[spikes[:,0].argsort()]

    else:
        # try loading
        spike_file = froot.parent / 'Regions' / (froot.stem + '_spikes.npz')
        if not reload and spike_file.exists():
            spikes = loadRegionSpikes(spike_file)
        else:
            if anat_file is None:
                raise ValueError("'anat_file' must be provided when 'output' = 'regions'")
            anat = loadAnatomyFile(anat_file)
            anat = anat[anat['rat'] == int(froot.stem[3:6])] # keep rat of interest, deduced from file name
            ids = np.unique(anat['region'])
            spikes = {id: {} for id in ids}
            unit_count = 0
            for id in ids:
                spikes[id]['e_group'] = {g: [] for g in np.unique(anat[anat['region'] == id]['electrode'])}
                sp = []
                for electrode in spikes[id]['e_group']:
                    if electrode in clu_spikes:
                        # label units
                        s = clu_spikes[electrode]
                        u, idx = np.unique(s[:,1:], axis=0, return_inverse=True)
                        sp.append(np.stack((s[:,0],idx+unit_count),axis=1))
                        spikes[id]['e_group'][electrode] = np.arange(unit_count,unit_count+len(u))
                        unit_count = unit_count + len(u)
                if len(sp):
                    sp = np.concatenate(sp)
                    spikes[id]['spikes'] = sp[sp[:,0].argsort()] # sort by time
                else:
                    spikes[id]['spikes'] = np.array(sp,ndmin=2)

            # save
            saveRegionSpikes(spike_file,spikes)

    return spikes


def loadAnatomyFile(file_path):
    # load .anat file, whose columns must be [rat, electrode, brain region] (comma separated)
    #
    # arguments:
    #     file_path    string = None, path to .anat file

    return np.genfromtxt(file_path,delimiter=",",comments="%",dtype=[("rat",int),("electrode",int),("region","U50")])


def loadRegionSpikes(file):
    spike_npz = np.load(file)
    spikes = {r[3:]: {'spikes': spike_npz[r]} for r in spike_npz.files if r.startswith('sp_')}
    for reg in spikes:
        spikes[reg]['e_group'] = {int(r.rsplit('_',1)[-1]) : spike_npz[r] for r in spike_npz.files if r.startswith(f'un_{reg}')}

    return spikes


def saveRegionSpikes(file,spikes):
    file = pathlib.Path(file)
    if not file.parent.exists():
        pathlib.Path.mkdir(file.parent)
    np.savez(file, **{f'sp_{r}': s['spikes'] for r, s in spikes.items()},
                   **{f'un_{r}_{g}': units for r, s in spikes.items() for g, units in s['e_group'].items()})

    return


def loadEventFile(filename: str, compact: bool = False):
    # load events from a .evt file
    #
    # arguments:
    #     filename    string, .evt file, each line must have format 'beginning of basename_event1_1'
    #     compact     bool = false, if true, return events as compact dictionary
    #
    # output:
    #     events      dict, keys are either 'times' and 'descriptions' or event names if 'compact' = True

    with open(filename,'r') as f:
        lines = f.read().splitlines()

    # extract first non-whitespace token
    times = [line.split()[0] for line in lines]
    times = np.array([float(t) / 1000.0 for t in times]) # convert to seconds

    # remove first token and following whitespace
    descriptions = []
    for line in lines:
        parts = line.split(maxsplit=1)
        descriptions.append(parts[1] if len(parts) > 1 else '')

    if compact:
        # group by events, description is of type 'beginning of basename_event1' or 'beginning of basename_event1_1'
        events = {} #collections.defaultdict(lambda: collections.defaultdict(list))
        for t, d in zip(times,descriptions):
            if " of " not in d:
                raise fmatoolbox.exceptions.FileFormatError(f"Unexpected format: '{d}'")
            phase, full_id = d.split(" of ",1)
            phase = phase.strip()
            full_id = full_id.strip()
            # remove trailing 'bis' and '_1'
            if full_id.endswith('bis'):
                full_id = full_id[:-3]
            parts = full_id.split('_')
            try:
                if parts[-1].isdigit():
                    part = parts[-2]
                else:
                    part = parts[-1]
            except:
                raise ValueError(f"Invalid event ID format: '{full_id}'")
            event_id = part
            # concatenate values
            if event_id in events:
                if phase in events[event_id]:
                    events[event_id][phase].append(t)
                else:
                    events[event_id][phase] = [t]
            else:
                events[event_id] = {phase : [t]}
    else:
        events = { 'time': times, 'description': descriptions }

    return events


def loadEvents(session:str, extra:str|list[str]):
    # load event files from a session
    #
    # arguments:
    #     session      string, path to session .xml file, event files must be in session folder
    #     extra        (:) string, extensions of other event files named <basename>.extra[i] to load,
    #                  can be 'subdir/extension' to load '.../basename/subdir/basename.extension'
    #
    # output:
    #     events       dict, keys are event names
    #     cat_names    (:) string, names of events contained in <basename>.cat.evt

    session = pathlib.Path(session)
    file_root = session.parent
    # accept single string input
    if isinstance(extra,str):
        extra = [extra]

    # load all *.evt files
    events = {}
    cat_names = []
    for file in file_root.glob("*.evt"):
        try:
            events.update(loadEventFile(file,compact=True))
        except fmatoolbox.exceptions.FileFormatError:
            pass
        if str(file)[-8:] == '.cat.evt':
            cat_names = list(events.keys())

    # load other file types
    for extension in extra:
        # handle subfolders, e.g., 'subfolder/name'
        e = pathlib.Path(extension).name
        p = pathlib.Path(extension).parent
        this_froot = file_root / p / session.stem
        if this_froot.with_suffix(f'.{e}.npz').exists():
            # load .npz file
            data = np.load(this_froot.with_suffix(f'.{e}.npz'))
            for d in data:
                events[d] = {f'col{i}': data[d][:,i] for i in range(data[d].shape[1])}
        elif this_froot.with_suffix(f'.{e}.events.mat').exists():
            # load .events.mat file
            events[e] = scipy.io.loadmat(this_froot.with_suffix(f'.{e}.events.mat'),simplify_cells=True)[e]
        else:
            # load text file
            data = np.loadtxt(this_froot.with_suffix(f'.{e}'),comments='%',delimiter=',',ndmin=2)
            events[e] = {f'col{i}': data[:,i] for i in range(data.shape[1])}

    return events, cat_names


def loadSpikeWaveforms(session: str):

    file_root = pathlib.Path(session).with_suffix('')
    data = scipy.io.loadmat(file_root.with_suffix('.cell_metrics.cellinfo.mat'),simplify_cells=True)['cell_metrics']
    waveforms = data['waveforms']
    # INSPECT struct TO FIND OUTPUT
    print('implement!')

    return


def loadLFP(session: str):

    print('implement!')

    return


def saveMatrix():
   # save matrix in standard FMAT format, prepending metadata header 

    print('implement!')

    return

# functions to run batch

def readBatchFile(file_path: str):
    # read batch file
    #
    # arguments:
    #     file_path    batch file
    #
    # output:
    #     sessions     DESCRIBE
    #     args         DESCRIBE
    
    sessions = []
    args = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except OSError:
        raise IOError(f"Unable to open {file_path}")

    for line in lines:

        # strip spaces, remove inline comments (anything after a %)
        line = line.strip().split('%',1)[0].strip()

        # split into words
        words = line.split()
        if not words:
            continue

        # first word is the session name
        sessions.append(words[0])

        # remaining words are arguments
        session_args = []
        for w in words[1:]:
            try:
                # evaluate numbers, lists, etc.
                value = ast.literal_eval(w)
            except (ValueError, SyntaxError):
                # if not a literal, keep as string
                value = w
            session_args.append(value)

        args.append(session_args)

    return sessions, args


def _batchWorker(payload):
    # private function to unpack payload and dispatch it to `func` inside runBatc
    i, func, session, args, extra_args, kwargs, seed = payload
    try:
        result = func(session,*args,*extra_args,**kwargs,**seed)
        return i, result
    except Exception as e:
        return -i-1, e


def runBatch(batch_file:str, func:Callable, args:list[list[Any]]=None, rnd_seed:str=None, kwargs:dict|list[dict]=None, ignore_args:bool=False,
             sessions:list[int]=None, parallel:bool|int=False, verbose:bool=True) -> tuple[list, ...]:
    # run a routine on multiple sessions
    #
    # arguments:
    #     batch_file     string, path to batch file
    #     func           function to call for each session, must take session path as first arg
    #     args           list of list = [[]], positional arguments for 'func', one per session or a single list for all
    #     rnd_seed       str = None, if given, spawn numpy random seeds, passed to 'func' as keyword argument, necessary to use np.random with 'parallel'
    #     kwargs         list of dict = [{}], keyword arguments for 'func', a dict per session or one for all
    #     ignore_args    bool = False, if True, ignore extra arguments from batch file
    #     sessions       (:) int = None, indices of session to process (default is all sessions from batch file)
    #     parallel       bool | int = False, parallelize calls of `func` with concurrent.futures
    #     verbose:       bool = True, log progress
    #        
    # output:
    #     variable outputs matching func's signature
    
    # parse batch file
    sessions_list, extra_args = readBatchFile(batch_file)
    if sessions is not None:
        if not isinstance(sessions, collections.abc.Iterable):
            sessions = [sessions]
        sessions_list = [sessions_list[i] for i in sessions]
        extra_args = [extra_args[i] for i in sessions]
    n_sessions = len(sessions_list)
    if ignore_args:
        extra_args = [[]] * n_sessions
    
    # validate optional arguments
    if args is None:
        args = [[]]
    if len(args) == 1:
        args = args * n_sessions
    elif len(args) != n_sessions:
        raise ValueError("Argument 'args' must contain one list per session")
    if rnd_seed is not None:
        seed_gen = np.random.SeedSequence()
        seeds = seed_gen.spawn(n_sessions)
        seeds = [{rnd_seed: s} for s in seeds]
    else:
        seeds = [{}] * n_sessions
    if kwargs is None:
        kwargs = [{}]
    elif isinstance(kwargs,dict):
        kwargs = [kwargs]
    if len(kwargs) == 1:
        kwargs = kwargs * n_sessions
    elif len(kwargs) != n_sessions:
        raise ValueError("Argument 'kwargs' must contain one dict per session")
    
    verbose and print(f"\nStarting Batch, {datetime.datetime.now()} \n")
    results = [None] * n_sessions
    errors = {}
    RED = "\033[31m"
    RESET = "\033[0m"

    # 1. Serial Batch
    if not parallel:
        for i, session in enumerate(sessions_list):
            verbose and print(f'Batch progress: {session}, {i+1} out of {n_sessions}')
            try:
                results[i] = func(session,*args[i],*extra_args[i],**kwargs[i],**seeds[i])
            except Exception as e:
                # log error to console
                errors[-i-1] = e
                print(f'Error in session {session}\n{str(e)}\nTraceback:')
                tb = e.__traceback__
                while tb:
                    fcode = tb.tb_frame.f_code
                    print(f'{RED}{fcode.co_filename}, line {tb.tb_lineno}, in {fcode.co_name}{RESET}')
                    tb = tb.tb_next
            verbose and print()

    # 2. Parallel Batch
    else:
        # pack arguments into payloads
        payloads = [(i,func,s,args[i],extra_args[i],kwargs[i],seeds[i]) for i, s in enumerate(sessions_list)]
        if parallel is True:
            parallel = None # to keep default max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
            for i, result in ex.map(_batchWorker, payloads):
                if i >= 0:
                    results[i] = result
                else:
                    errors[i] = result
        # log errors to console
        for i, e in errors.items():
            print(f'Error in session {sessions_list[-i-1]} ({-i-1})\n{str(e)}\nTraceback:')
            tb = e.__traceback__
            while tb: # NOT WORKING FOR NOW
                fcode = tb.tb_frame.f_code
                print(f'{RED}{fcode.co_filename}, line {tb.tb_lineno}, in {fcode.co_name}{RESET}')
                tb = tb.tb_next

    # determine output number (expected to be constant across func calls)
    n_outs = None
    outputs = [None] * n_sessions
    for result in results:
        if result is not None:
            if isinstance(result,tuple):
                n_outs = len(result)
                outputs = [outputs.copy() for _ in range(n_outs)]
            else:
                n_outs = 0  # 0 marks the single output case
            break
    # assign outputs
    if n_outs is not None:
        for i in range(n_sessions):
            if results[i] is not None:
                if n_outs == 0:
                    outputs[i] = results[i]
                else:
                    for j in range(n_outs):
                        outputs[j][i] = results[i][j]

    verbose and print(f'Batch completed with {len(errors)} errors, {datetime.datetime.now()}')
    
    return tuple(outputs) if n_outs else outputs