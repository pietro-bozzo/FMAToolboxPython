''' Session data handling functions for FMAToolbox '''

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


def loadAnatomyFile(file_path):
    # load .anat file, whose columns must be [rat, electrode, brain region] (comma separated)
    #
    # arguments:
    #     file_path    string = None, path to .anat file

    return np.genfromtxt(file_path,delimiter=",",comments="%",dtype=[("rat",int),("electrode",int),("region","U50")])


def loadSpikeTimes(session:str, output:str='dict', anat_file:str=None, return_elec:bool=False, return_loc:bool=False):
    # load spikes from a session
    #
    # arguments:
    #     session         string, path to session .xml file, spike files must be in session folder
    #     output          string = None, determines output type, can be:
    #                       'dict', dictionary of spike times per unit
    #                       'compact', (:,2) array of [timestamps, unit ids]
    #                       'full', (:,2) array of [timestamps, electrode groups, clusters]
    #                       'regions', dict of {region id: dict}, entries are {'spikes': region spikes, 'units': region units}
    #     return_elec      bool = False, if true, return cluster_loc
    #     return_loc       bool = False, if true, return cluster_loc
    #
    # output:
    #     spikes          (see 'output')
    #     electrodes      (n) float, optional, electrode id per unit
    #     cluster_loc     (n) float, optional, index of max spike-amplitude cluster per unit

    if output not in ['dict','compact','full','regions']:
        raise ValueError("'output' must be 'dict', 'compact', 'full', or 'regions'")

    file_root = pathlib.Path(session).with_suffix('')
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
        ids = np.repeat(unit_id,[len(s) for s in spikes])
        spikes = np.stack((np.concatenate(spikes),ids),axis=1)
        spikes = spikes[spikes[:,0].argsort()]

    elif output == 'compact':
        electrodes = np.repeat(electrode_id,[len(s) for s in spikes])
        cluster_id = data['cluID'] # min is 2, as 0 and 1 are excluded clusters?
        clusters = np.repeat(cluster_id,[len(s) for s in spikes])
        spikes = np.stack((np.concatenate(spikes),electrodes,clusters),axis=1)
        spikes = spikes[spikes[:,0].argsort()]

    else:
        # try loading
        spike_file = file_root.parent / 'Regions' / (file_root.stem + '_spikes.npz')
        if spike_file.exists():
            spike_npz = np.load(spike_file)
            spikes = {r[3:]: {'spikes': spike_npz[r]} for r in spike_npz.files if r.startswith('sp_')}
            for r in spikes:
                spikes[r]['units'] = spike_npz[f'un_{r}']
        else:
            spikes_dict = dict(zip(unit_id,spikes))
            if anat_file is None:
                raise ValueError("'anat_file' must be provided when 'dict' = 'regions'")
            anat = loadAnatomyFile(anat_file)
            anat = anat[anat['rat'] == int(file_root.stem[3:6])] # keep rat of interest, deduced from file name
            ids = np.unique(anat['region'])
            spikes = {id : {} for id in ids}
            for id in ids:
                spikes[id]['units'] = []
                s = []
                for electrode in anat[anat['region'] == id]['electrode']:
                    electrode_units = unit_id[electrode_id == electrode]
                    spikes[id]['units'].append(electrode_units)
                    for unit in electrode_units:
                        s.append(np.array([spikes_dict[unit], [unit] * spikes_dict[unit].size]).T)
                # assing spikes, sorted by time
                s = np.concatenate(s)
                spikes[id]['spikes'] = s[s[:, 0].argsort()]
                spikes[id]['units'] = np.concatenate(spikes[id]['units'])
            # save
            if not spike_file.parent.exists():
                pathlib.Path.mkdir(spike_file.parent)
            np.savez(spike_file,**{f'sp_{r}': s['spikes'] for r, s in spikes.items()},**{f'un_{r}': s['units'] for r, s in spikes.items()})

    if not return_elec and not return_loc:
        return spikes
    out = (spikes, electrode_id, cluster_loc) # prepare tuple to return requested outputs
    return out[:2+return_loc:2-return_elec]


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
        events = collections.defaultdict(lambda: collections.defaultdict(list))
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
            events[event_id][phase].append(t)
    else:
        events = { 'time': times, 'description': descriptions }

    return events


def loadEvents(session: str, extra: list[str]):
    # load event files from a session
    #
    # arguments:
    #     session    string, path to session .xml file, event files must be in session folder
    #     extra      (:) string, extensions of other event files named basename.extra[i] to load as text files,
    #                can be 'subdir/extension' to load '.../basename/subdir/basename.extension'
    #
    # output:
    #     events     dict, keys are event names

    session = pathlib.Path(session)
    file_root = session.parent
    # accept single string input
    if isinstance(extra,str):
        extra = [extra]

    # load all *.evt files
    events = {}
    for file in file_root.glob("*.evt"):
        try:
            this_events = loadEventFile(file,compact=True)
        except fmatoolbox.exceptions.FileFormatError:
            this_events = {}
        for event in this_events.keys():
            # MUST ENFORCE THAT BEGINNING IS FIRST AND END IS SECOND!!
            if event in events:
                events[event] = np.concatenate((events[event],np.stack([t for t in this_events[event].values()],axis=1)))
            else:
                events[event] = np.stack([t for t in this_events[event].values()],axis=1)

    # load other file types as text files
    for extension in extra:
        e = pathlib.Path(extension).name
        p = pathlib.Path(extension).parent
        file_path = file_root / p / (session.with_suffix('').name +'.' + e)
        events[e] = np.loadtxt(file_path,comments='%',delimiter=',')

    return events


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
    #     rnd_seed       str = None, if given, spawn numpy random seeds, passed to 'func' as keyword argument
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