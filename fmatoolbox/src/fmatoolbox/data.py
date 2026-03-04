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


def loadSpikeTimes(session: str, output: str = 'dict', return_elec: bool = False, return_loc: bool = False):
    # load spikes from a session
    #
    # arguments:
    #     session         string, path to session .xml file, spike files must be in session folder
    #     output          string = None, determines output type, can be:
    #                       'dict', dictionary of spikes per unit
    #                       'compact', (:,2) array of [timestamps, unit ids]
    #                       'full', (:,2) array of [timestamps, electrode groups, clusters]
    #     return_elec      bool = False, if true, return cluster_loc
    #     return_loc       bool = False, if true, return cluster_loc
    #
    # output:
    #     spikes          (see output)
    #     electrodes      (n) float, optional, electrode id per unit
    #     cluster_loc     (n) float, optional, index of max spike-amplitude cluster per unit

    if output not in ['dict','compact','full']:
        raise(ValueError('\'output\' must be \'dict\', \'compact\' or \'full\''))

    file_root = pathlib.Path(session).with_suffix('')
    data = scipy.io.loadmat(file_root.with_suffix('.cell_metrics.cellinfo.mat'),simplify_cells=True)['cell_metrics']
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

    else:
        electrodes = np.repeat(electrode_id,[len(s) for s in spikes])
        cluster_id = data['cluID'] # min is 2, as 0 and 1 are excluded clusters?
        clusters = np.repeat(cluster_id,[len(s) for s in spikes])
        spikes = np.stack((np.concatenate(spikes),electrodes,clusters),axis=1)
        spikes = spikes[spikes[:,0].argsort()]

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
    times = [re.sub(r'([^ \t]*).*',r'\1',line,count=1) for line in lines]
    times = np.array([float(t) / 1000.0 for t in times]) # convert to seconds

    # remove first token and following whitespace
    descriptions = [re.sub(r'[^ \t]*[ \t]*', '', line, count=1) for line in lines]

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


def runBatch(batch_file: str, func: Callable, args: list[list[Any]] = None, kwargs: list[dict] = None, ignore_args: bool = False,
             sessions: list[int] = None, verbose: bool = True) -> tuple[list, ...]:
    # run a routine on multiple sessions
    #
    # arguments:
    #     batch_file     string, path to batch file
    #     func           function to call for each session, must take session path as first arg
    #     args           list of list = [[]], positional arguments for 'func', one per session or a single list for all
    #     kwargs         list of dict = [{}], keyword arguments for 'func', a dict per session or one for all
    #     ignore_args    bool = False, if True, ignore extra args from batch file
    #     sessions       (:) int = None, indices of session to process (default is all sessions from batch file)
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
    if kwargs is None:
        kwargs = [{}]
    elif isinstance(kwargs,dict):
        kwargs = [kwargs]
    if len(kwargs) == 1:
        kwargs = kwargs * n_sessions
    elif len(kwargs) != n_sessions:
        raise ValueError("Argument 'kwargs' must contain one dict per session")
    
    verbose and print(f"\nStarting Batch, {datetime.datetime.now()} \n")
    n_outs = None
    outputs = [None] * n_sessions
    errors = 0
    for i, session in enumerate(sessions_list):

        verbose and print(f'Batch progress: {session}, {i+1} out of {n_sessions}')
        
        try:
            result = func(session,*args[i],*extra_args[i],**kwargs[i])

            # allocate list for outputs
            if n_outs is None:
                if isinstance(result, tuple):
                    n_outs = len(result)
                    outputs = [outputs.copy() for _ in range(n_outs)]
                else:
                    n_outs = 0 # 0 marks the single output case

            # assign outputs
            if n_outs == 0:
                outputs[i] = result
            else:
                for j in range(n_outs):
                    outputs[j][i] = result[j]
                
        except Exception as e:
            errors += 1
            print(f'Error in session {session}')
            print(f'{str(e)}')
            print('Stacktrace:')
            traceback.print_exc()
            print()
        
        verbose and print()
    
    verbose and print(f'Batch completed with {errors} errors')
    
    return tuple(outputs) if n_outs else outputs