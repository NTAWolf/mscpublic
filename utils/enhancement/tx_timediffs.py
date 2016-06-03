import os
import pickle
import multiprocessing

import pandas as pd
import numpy as np

import utils.data as ud

import tbtools.iter as tbiter

def worker(uid_gr):
    """Takes a groupby next() on uid
    Returns a dict where keys are tuples of tx points,
        and values are times seen between those tx points.
    """
    uid,gr = uid_gr
    times = dict()
    for i in range(len(gr)):
        for k in range(i+1, len(gr)):
            key = (gr.Tx.iloc[i],gr.Tx.iloc[k])
            if not key in times:
                times[key] = []
            times[key].append( (gr.index[k] - gr.index[i]) / pd.to_timedelta('1 s') )

    return times

def leader(bbh):
    """Takes an enhanced bitbushist
    Returns a aggregated dictionary, where the
        keys are tx pairs
        values are lists of times seen between those tx pairs
    """
    with multiprocessing.Pool() as p:
        timeslist = p.map(worker,
                          tbiter.IProgressBar(bbh.groupby('uid'),
                                              bbh.uid.nunique()))
    print('Completed map')
    print()
    moab = dict()
    for t in tbiter.IProgressBar(timeslist):
        for k in t:
            if not k in moab:
                moab[k] = t[k]
            else:
                moab[k].extend(t[k])
    print('Completed reduce')
    print()
    return moab

def _map_and_store(bbh):
    """bbh is an enhanced bitbushist
    returns a DataFrame with cols from, to, and times
        (and saves it on disk)
    """
    times = leader(bbh)
    path = os.path.join(ud.Paths.enhanced, 'tx_timediffs.csv')
    df = None
    for k,v in tbiter.IProgressBar(times.items(), len(times)):
        d = pd.DataFrame({'from':k[0], 'to':k[1], 'times':v})
        if df is None:
            df = d
        else:
            df = df.append(d)
    df.to_csv(path, index=False)
    print('saved list of times as DataFrame to {}'.format(path))
    return df

def run(bbh=None):
    if bbh is None:
        bbh = ud.enhanced.get('bitbushist')

    times = _map_and_store(bbh)
    _reduce_and_store()

def _reduce_and_store(times=None):
    path = os.path.join(ud.Paths.enhanced, 'tx_timediffs.csv')
    times = pd.read_csv(path)

    df = times.groupby(['from', 'to']).median()

    path = os.path.join(ud.Paths.enhanced, 'median_tx_timediffs.csv')
    df.to_csv(path, index=True)
    print('saved medians to {}'.format(path))


