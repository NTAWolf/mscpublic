import multiprocessing
import os

import pandas as pd
import numpy as np

import utils.data as ud
import tbtools.iter as tbiter

IN_L1 = (1, 2)
IN_L2 = (7, 8)
IN_EMERGENCY = (2,8)
IN_NORMAL = (1,7)
# Assume out is always from the line last entered
OUT = (15, 16, 3)

def clear_duplicates(statechanges):
    sc = statechanges
    # Forward pass: Remove duplicates 1s
    state = 0
    for i,s in sc.items():
         if state == 0:
            state = s
            continue
         if state == 1:
            if s == 1:
                sc[i] = 0
            elif s == -1:
                 state = 0

    # Backward pass: Remove duplicate -1s
    state = 0
    for i,s in reversed(list(sc.items())):
        if state == 0:
            state = s
            continue
        if state == -1:
            if s == -1:
                sc[i] = 0
            elif s == 1:
                state = 0

    return sc

def handle_line_attribution(states):
    cl = states['change_line']
    cl[states['change'] != 1] = np.nan
    cl = cl.ffill()
    cl[states['change'] == 0] = 0
    states['change_line'] = cl.astype(int)

def worker(uid_gr):
    """Parallel-friendly workhorse for assigning reinspection
    change to bitbushist.
    """
    uid,gr = uid_gr

    enter1 = np.in1d(gr.Tx.values, IN_L1).astype(int)
    enter2 = np.in1d(gr.Tx.values, IN_L2).astype(int)
    leave = np.in1d(gr.Tx.values, OUT).astype(int)

    states = pd.DataFrame({'change':(enter1+enter2 - leave),
                           'change_line':enter1 + 2*enter2},
                           index=gr.index)

    states['change'] = clear_duplicates(states['change'])
    handle_line_attribution(states)

    assert all(states.groupby('change_line')['change'].sum() == 0),\
           "Problems with uid {}".format(uid)
    assert all(states.cumsum() >= 0),\
           "Problems with uid {}".format(uid)

    # print('silent assertions')

    return states

def encode_change_causes(bbh):
    bbh['reinspection_normal'] = np.in1d(bbh.Tx, IN_NORMAL)
    bbh['reinspection_emergency'] = np.in1d(bbh.Tx, IN_EMERGENCY)

def leader(bbh):
    bbh = bbh.reset_index().sort_values('Timestamp')
    nuids = bbh.uid.nunique()
    # ric = map(worker, tbiter.IProgressBar(bbh.groupby('uid'), nuids))
    with multiprocessing.Pool() as p:
        ric = p.map(worker, tbiter.IProgressBar(bbh.groupby('uid'), nuids))
    res = pd.concat(ric, axis=0).sort_index()

    bbh['reinspection_change'] = res['change']
    bbh['reinspection_change_line'] = res['change_line']
    encode_change_causes(bbh)

    return bbh.set_index('Timestamp').sort_index()

def run():
    df = ud.enhanced.get('bitbushist')
    df = leader(df)
    path = os.path.join(ud.Paths.enhanced, 'bitbushist.csv')
    df.to_csv(path)
    print('Result saved in {}'.format(path))
