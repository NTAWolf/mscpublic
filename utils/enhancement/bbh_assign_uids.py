import itertools
import multiprocessing
import os

import pandas as pd
import numpy as np

from .. import data as ud
from ..data.paths import Paths

import tbtools.strings as tbstr
import tbtools.iter as tbiter

def id_by_lagged_timedifference(times, max_diff):
    """times is a pd.Series of datetimes
    max_diff is a timedelta. It is the minimum time between
    consecutive datetimes to consider them as different ids.
    """
    diffs = pd.Series(times.values[1:] - times.values[:-1])
    diffs = diffs > max_diff
    s = np.ones(len(times), dtype=int)
    s[1:] += diffs.cumsum().values

    return s

def get_uid_series_worker(hid_gr_maxinterval, max_interval=pd.to_timedelta('3 hours')):
    """hid_gr_maxinterval is a tuple:
        (HangerID, bbh.group), max_interval
    where the first subtuple is from bbh.reset_index().groupby('HangerID')
    and the max_interval is a timedelta.
    """
    (hid, gr), max_interval = hid_gr_maxinterval
    uids = pd.Series(-1, index=gr.index)

    arr = id_by_lagged_timedifference(gr.Timestamp, max_interval)
    uids[gr.index] = arr

    return uids

def get_uid_series_collector(bbh, max_interval='3 hours'):
    max_interval = pd.to_timedelta(max_interval)
    pool = multiprocessing.Pool()

    bbh = bbh.reset_index()
    groups = bbh.groupby('HangerID')
    arg = zip(groups, itertools.repeat(max_interval))
    pb = tbiter.IProgressBar(arg, len(groups))
    res = pool.map(get_uid_series_worker, pb)

    for i in range(1, len(res)):
        curmax = res[i-1].max()
        res[i] += curmax

    ser = pd.concat(res, axis=0).sort_index()
    return ser

def assign_uids(bbh, max_interval='7 hours', target_col='uid'):
    ser = get_uid_series_collector(bbh)
    bbh[target_col] = ser.values
    return bbh

# def assign_uids(bbh, max_interval='7 hours'):
#     max_interval = pd.to_timedelta(max_interval)
#     uids = pd.Series(-1, index=range(len(bbh)))
#     idx = bbh.index
#     bbh = bbh.reset_index()

#     curr_id = 0
#     for hid, gr in tbiter.IProgressBar(bbh.groupby('HangerID')):
#         local_ids = id_by_lagged_timedifference(gr.Timestamp, max_interval)
#         local_ids += curr_id
#         curr_id = local_ids.max()
#         uids[gr.index] = local_ids

#     return pd.Series(uids.values, index=idx)

def run():
    print('Reading bitbushist')
    with tbstr.indent():
        bbh = ud.raw.get('bbh')

    print('Assigning uids')
    with tbstr.indent():
        bbh = assign_uids(bbh)
        # uids = assign_uids(bbh)
        # bbh['uid'] = uids

    path = os.path.join(Paths.enhanced, 'bitbushist.csv')
    print('Store in {}'.format(path))
    bbh.to_csv(path)
