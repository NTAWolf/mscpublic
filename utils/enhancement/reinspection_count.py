import numpy as np
import pandas as pd

from tbtools.iter import IProgressBar

REINSPECTION_TX_IN = set((1, 2, 7, 8))
REINSPECTION_TX_OUT = set((15, 16, 3))
REINSPECTION_TX_DEFINITELY_OUT = set((4, 5, 9, 10))

def get_reinspection_current_count(bbh):
    """Counts the number of hangers inside the reinspection
    at every timestep from a BitBusHist dataframe.

    Returns a three-tuple:
        counter: Series, index like bbh, values are number of carcasses
            in reinspection at the given time.
        irregulars: list of HangerIDs that do not conform to expectations
        leftovers: list of HangerIDs that are not registered as leaving
            the reinspection
    """
    inside = set()

    counter = pd.Series(np.zeros(len(bbh)), index=bbh.index)
    irregulars = []

    for timestamp, (uid, tx) in IProgressBar(bbh[['uids', 'Tx']].iterrows(),
                                             len(bbh)):
        if uid in inside:
            if tx in REINSPECTION_TX_OUT:
                inside.remove(uid)
            elif tx in REINSPECTION_TX_DEFINITELY_OUT:
                irregulars.append(uid)
                inside.remove(uid)
        elif tx in REINSPECTION_TX_IN:
            inside.add(uid)

        counter[timestamp] = len(inside)

    return counter, irregulars, list(inside)

# Use tuples, as they work with np.in1d
REINSPECTION_TX_IN = (1, 2, 7, 8)
REINSPECTION_TX_OUT = (15, 16, 3)
REINSPECTION_TX_DEFINITELY_OUT = (4, 5, 9, 10, 6, 11, 17)

def _reinspection_count_worker(hangerid_group):
    hid, vals = hangerid_group
    
    pass

def get_reinspection_current_count(bbh):
    """Counts the number of hangers inside the reinspection
    at every timestep from a BitBusHist dataframe.

    Returns a three-tuple:
        counter: Series, index like bbh, values are number of carcasses
            in reinspection at the given time.
        irregulars: list of uids that do not conform to expectations
        leftovers: list of uids that are not registered as leaving
            the reinspection
    """

    # Curious note to self:
    # It seems that np.in1d is about 3 times faster than pd.ser.isin

    bbh = bbh.sort_index().reset_index()


    inside = set()
    irregulars = []

    bbh['movements'] = np.in1d(bbh.Tx.values, REINSPECTION_TX_IN).astype(int)\
                     - np.in1d(bbh.Tx.values, REINSPECTION_TX_OUT).astype(int)

    n_uids = len(bbh.uids.unique())
    inspect = []
    for uid, vals in IProgressBar(bbh.groupby('uids'), n_uids):
        s = vals.movements.sum()
        if s != 0:
            inspect.append(uid)

    # Use multiprocessing.Pool.map here to examine the uids
    # in `inspect`



    leaving = bbh.Tx.isin(REINSPECTION_TX_OUT)
    moves = bbh.Tx.isin(REINSPECTION_TX_IN) - leaving
    defleft = bbh.Tx.isin(REINSPECTION_TX_DEFINITELY_OUT)


    for (uid, leave), (uid, left) in \
        zip(pd.groupby(leaving, by=bbh.uids),
            pd.groupby(defleft, by=bbh.uids)):
        pass # TODO

    counter = (entering - leaving).cumsum()


