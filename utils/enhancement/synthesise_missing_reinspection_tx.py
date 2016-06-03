# TODO synth when first entry is exit from reinspection

import itertools
import multiprocessing
import os

import pandas as pd
import numpy as np

import utils.data as ud

import tbtools.iter as tbiter

def median_time(source, destination):
    """Both inputs are tx points.
    Returns the median time seen in data from
    source to destination.
    """
    td = ud.enhanced.get('median', silent=False)
    res = td[(td['from']==source) & (td['to']==destination)]
    if len(res) == 0:
        return pd.tslib.NaT
    res= res.median_seconds.values[0]
    return pd.to_timedelta('{} seconds'.format(res))

def _median_state(truetime, target_tx, tx_before, before,
                    tx_after=None, after=None):
    if tx_after is None:
        return 'n/a'
    delta_b = median_time(tx_before, target_tx)
    delta_a = median_time(target_tx, tx_after)
    state = ''
    if (after - delta_a) < before:
        state += 'a-da < b;'
    if (before + delta_b) > after:
        state += 'b+db > a'
    if len(state) == 0:
        state = 'within'
    return state

def datetime_mean(dt1, dt2):
    """Returns the mean of two datetimes.
    """
    to_middle = (dt2 - dt1)/2
    return dt1 + to_middle

def weighted_mean(a,b,wa,wb):
    """Returns the weighted mean of a and b,
    with weights wa and wb, respectively.
    Works for Datetimes too!
    """
    diff = b-a
    w = wb / (wa+wb)
    res = a + w*diff
    return res

def guess_interpolate(target_tx, tx_before, before,
                    tx_after=None, after=None):
    """target_tx, tx_before, tx_after are tx values
    before, after are pd.Timestamp
    Returns a guess at what time target_tx must have occured.
    """
    delta_b = median_time(tx_before, target_tx)
    if after is None:
        return before + delta_b

    delta_a = median_time(target_tx, tx_after)
    # Convert timedeltas to floats with seconds as their units
    # Also, notice that we flip so delta_b becomes the weight for a
    # and vice versa.
    s = pd.to_timedelta('1 s')
    weight_a = delta_b/s
    weight_b = delta_a/s
    res = weighted_mean(before, after, weight_b, weight_a)
    return pd.Timestamp(res)

def guess_mean(target_tx, tx_before, before,
                    tx_after=None, after=None):
    delta_b = median_time(tx_before, target_tx)
    b = before + delta_b
    if after is None:
        return b

    delta_a = median_time(target_tx, tx_after)
    a = after - delta_a
    return datetime_mean(a,b)

def synth(surroundings, target_tx):
    """surroundings are one or two bbh-like rows
    target_tx is the tx value of the new row
        to be synthesised.

    This assumes the two surroundings rows (if there are two)
        have the same parameters for HangerID, uid, and others.

    Returns a single synthesised row.
    """
    res = surroundings.iloc[0].copy()
    res['Tx'] = target_tx
    res['synthesised'] = 1

    ts = surroundings.Timestamp.values
    tx = surroundings.Tx.values
    t_before = median_time(tx[0], target_tx)
    if len(surroundings) == 2:
        guess = guess_interpolate(target_tx, tx[0], ts[0], tx[1], ts[1])
        # Assert that bestguess is between surroundings?
        # Disregard target surrounding if it is tx3?
    elif len(surroundings) == 1:
        guess = guess_interpolate(target_tx, tx[0], ts[0])
    else:
        raise AssertionError('surroundings of bad length. '
                             'shape {}'.format(surroundings.shape))

    res['Timestamp'] = guess

    return res

def synth_before(point, target_tx):
    """Handles synthesis of a target_tx BEFORE
    a reference point (bitbushist record)
    """
    res = point.copy()
    res['Tx'] = target_tx
    res['synthesised'] = 1

    ts = point['Timestamp']
    tx = point['Tx']
    mt = median_time(target_tx, tx)
    res['Timestamp'] -= mt

    return res

def get_synth_target(tx1, tx2=None):
    if tx1 is None and tx2 is not None:
        if tx2 == 15:
            return 1
        if tx2 == 16:
            return 7
        return None

    if tx1 in (1,2) and tx2 in (4,5,17, None):
        # synth L1 leave: 15
        return 15
    if tx1 in (7,8) and tx2 in (9,10,17, None):
        # synth L2 leave: 16
        return 16
    if tx2 == 3:
        if tx1 in (13,14):
            return 1
        if tx1 == 12:
            return 7

    if tx2 == 15:
        if tx1 in (21,22,23,13,14):
            # synth L1 normal enter: 1
            return 1
        if tx1 in (4,5,6):
            # synth L1 emergency enter: 2
            return 2

    if tx2 == 16:
        if tx1 in (21,22,23,12):
            # synth L2 normal enter: 7
            return 7
        if tx1 in (9,10,11):
            # synth L2 emergency enter: 8
            return 8

    return None

def synth_worker(uidgroup):
    uid, gr = uidgroup
    synths = []

    s = gr.iloc[0]
    synthtarget = get_synth_target(None, s.Tx)
    if synthtarget is not None:
        res = synth_before(s, synthtarget)
        synths.append(res)

    for i in range(len(gr)):
        s = gr.iloc[i:i+2]
        synthtarget = get_synth_target(*s.Tx.values)
        if synthtarget is not None:
            res = synth(s, synthtarget)
            synths.append(res)

    return synths

def leader(bbh=None):
    if bbh is None:
        bbh = ud.enhanced.get('bitbushist')

    bbh = bbh.reset_index().sort_values('Timestamp')
    if 'synthesised' in bbh:
        bbh = bbh[bbh['synthesised'] == 0]
    else:
        bbh['synthesised'] = 0
    synths = map(synth_worker,
                 tbiter.IProgressBar(bbh.groupby('uid'),
                                     bbh.uid.nunique()))
    # with multiprocessing.Pool() as p:
    #     synths = p.map(synth_worker,
    #                    tbiter.IProgressBar(bbh.groupby('uid'),
    #                                        bbh.uid.nunique()))
    synths = list(itertools.chain(*synths))
    bbh = bbh.append(synths, ignore_index=True)

    return bbh.set_index('Timestamp').sort_index()

def run():
    bbh = ud.enhanced.get('bitbushist')
    if 'synthesised' in bbh:
        if input('bitbushist already has synthesised '
                 'values. Proceed and destroy them?') == 'n':
            return
    newbbh = leader(bbh)
    path = os.path.join(ud.Paths.enhanced, 'bitbushist.csv')
    newbbh.to_csv(path)
    print('Result saved in {}'.format(path))

#
# Evaluation
#

# Extracting segments for development purposes
# Saved in /home/ntawolf/Speciale/data/dev/txsegments.csv

def get_segments(bbh):
    with multiprocessing.Pool() as p:
        res = p.map(segments_extractor, bbh.groupby('uid'))
    res = [x for x in res if x is not None]
    return pd.concat(res, ignore_index=True)

def segments_extractor(uid_gr):
    uid,gr = uid_gr
    gr = gr.reset_index()
    holdouts = gr.index[gr.Tx.isin([1,2,7,8,3,15,16])]
    if len(holdouts) == 0:
        return None

    tx = gr.Tx
    ts = gr.Timestamp

    df = pd.DataFrame({
        'ttx': tx[holdouts],
        't': ts[holdouts],
        'btx': [tx[i-1] if i>=1 else None for i in holdouts],
        'b': [ts[i-1] if i>=1 else pd.tslib.NaT for i in holdouts],
        'atx': [tx[i+1] if (i<len(gr)-1) else None for i in holdouts],
        'a': [ts[i+1] if (i<len(gr)-1) else pd.tslib.NaT for i in holdouts],
        'uid':uid,
    })

    df['amedian'] = list(map(median_time, df.ttx, df.atx))
    df['bmedian'] = list(map(median_time, df.btx, df.ttx))

    return df

# Slow-iteration development stuff
# Not to be used anymore.

def within_sample_error(bbh):
    """Diagnostic tool. Will evaluate guess_timestamp's
    performance on all existing reinspection tx records, and
    return a loooong array containing all the errors,
    signed and in seconds.
    """
    # bbh = bbh.reset_index()

    with multiprocessing.Pool() as p:
        errors_keys_g2states_medianstates = p.map(error_worker, bbh.groupby('uid'))
    # errors = list(map(error_worker, bbh.groupby('uid')))

    errors_keys_g2states_medianstates = [x for x in errors_keys_g2states_medianstates if len(x)>0]
    errors1, errors2, keys, g2states, medianstates = zip(*errors_keys_g2states_medianstates)
    errors1 = np.concatenate(errors1)
    errors2 = np.concatenate(errors2)
    keys = [x for y in keys for x in y]
    g2states = [x for y in g2states for x in y]
    medianstates = [x for y in medianstates for x in y]

    return errors1, errors2, keys, g2states, medianstates


def error_worker(uid_gr):
    uid,gr = uid_gr
    gr = gr.reset_index()
    holdouts = gr.index[gr.Tx.isin([1,2,7,8,3,15,16])]
    if len(holdouts) == 0:
        return holdouts
    if holdouts[0] == 0:
        # We can't look back from the first one. It is not really handled yet.
        holdouts = holdouts[1:]

    tx = gr.Tx
    ts = gr.Timestamp

    output = pd.DataFrame({
        'errormm':0.,
        'errorint':0.,
        'segment':None,
        'before':pd.to_datetime('')
        })
    errors1 = np.zeros(len(holdouts))
    errors2 = np.zeros(len(holdouts))
    keys = []
    g2states = []
    medianstates = []

    s = np.timedelta64(1,'s')

    for k,i in enumerate(holdouts):
        if i+1 < len(gr):
            key = (tx[i-1], tx[i], tx[i+1])
            guess1 = guess_interpolate(tx[i], tx[i-1], ts[i-1], tx[i+1], ts[i+1])
            guess2 = guess_mean(tx[i], tx[i-1], ts[i-1], tx[i+1], ts[i+1])
            ms = _median_state(ts[i], tx[i], tx[i-1], ts[i-1], tx[i+1], ts[i+1])
            if guess2 < ts[i-1]:
                g2state = 'before'
            elif guess2 > ts[i+1]:
                g2state = 'after'
            else:
                g2state = 'within'
        else:
            key = (tx[i-1], tx[i])
            guess1 = guess_interpolate(tx[i], tx[i-1], ts[i-1])
            guess2 = guess_mean(tx[i], tx[i-1], ts[i-1])
            g2state = 'n/a'
            ms = 'n/a'
        actual = ts[i]
        error1 = (actual - guess1) / s
        error2 = (actual - guess2) / s
        errors1[k] = error1
        errors2[k] = error2
        keys.append(key)
        g2states.append(g2state)
        medianstates.append(ms)

    return errors1, errors2, keys, g2states, medianstates