import functools

import pandas as pd
import numpy as np

import utils.data as ud

import tbtools.func as tbfunc
import tbtools.iter as tbiter
import tbtools.panda as tbpd


_ensure_bbh = tbfunc.ensure_kwarg('bbh', ud.enhanced.get, 'bitbushist')
_ensure_alm = tbfunc.ensure_kwarg('alm', ud.raw.get, 'almhist')

@_ensure_bbh
def throughput(tx, bbh, window='5 min'):
    """
    Returns the throughput at Tx point tx for the given
    bitbushist (or the full period), with the given window.
    """
    return bbh[bbh.Tx == tx].uid.resample(window).count()

def sample_C(sample_times, lag, line):
    """Samples the value of delta C at the given sample times,
    with the given lag, for the given line, with the given
    window.
    """
    c = C(line)
    c = c.asof(sample_times - pd.to_timedelta(lag))
    c.index = sample_times
    c.name += ' L={}'.format(lag)
    return c

@tbfunc.persistent_lru_cache(2*12, ud.paths.Paths.cache)
def C(line, lag=None):
    """
    Returns the reinspection count estimate for every point
    in the full timerange where a change has occurred.

    lag can be a string like '5 min'.
    """
    dc = delta_C(line)
    c = dc.cumsum().astype(int)
    c.name = 'C_{}'.format(line)
    if lag is not None:
        c = apply_lag(c, lag, update_name=True)
    return c

def sample_delta_C(sample_times, lag, line, window):
    """Samples the value of delta C at the given sample times,
    with the given lag, for the given line, with the given
    window.
    """
    dc = delta_C(line, mode='+', window=window)
    dc = dc.asof(sample_times - pd.to_timedelta(lag))
    dc.index = sample_times
    dc.name += ' L={}'.format(lag)
    return dc

@tbfunc.persistent_lru_cache(2*30, ud.paths.Paths.cache)
@_ensure_bbh
def delta_C(line, bbh, mode='', frequency='4 s',
            window=None, lag=None):
    """
    Returns the change in reinspection count for the
        given line. NB: Index is limited to `frequency`-sized
        gaps, but usually has more distance between entries;
        nans are discarded.

    mode is {'+'|'-'|''}.
        + means to only use the positive entries of ΔC
        - means to only use the negative entries of ΔC
        Anything else means to use both positive and
            negative entries of ΔC

    frequency is the resample frequency. It defaults to 4 seconds,
        which is the value to, on average, correctly represent
        the most change-dense minute in the whole dataset
        (which is at 13 changes/min)

    window and lag are strings like '5 min', used to get \Delta C
        as a summed window over some time, and/or as a lagged version.
    """
    rc = bbh.reinspection_change[bbh.reinspection_change_line == line]
    name = 'ΔC'
    if mode == '+':
        name = 'Δ+C'
        rc = rc[rc > 0]
    elif mode == '-':
        name = 'Δ-C'
        rc = rc[rc < 0]

    dc = rc.resample(frequency).sum().dropna()
    dc.name = '{}_{}'.format(name, line)

    if lag is not None:
        dc = apply_lag(dc, lag, update_name=True)
    if window is not None:
        dc = apply_summing_window(dc, window, resolution=frequency,
                                  update_name=True)
        # Keep only indices where a change happpens
        dc = dc[dc.diff() != 0]

    return dc

def sample_R(sample_times, lag, line, n=10):
    """Samples the value of R at the given sample times,
    with the given lag, for the given line, with the given
    number of carcasses-window n.
    """
    r = R(line, n)
    r = r.asof(sample_times - pd.to_timedelta(lag))
    r.index = sample_times
    r.name += ' L={}'.format(lag)
    return r

@tbfunc.persistent_lru_cache(2*20, ud.paths.Paths.cache)
@_ensure_bbh
def R(line, n, bbh, lag=None, emergency=False):
    """Returns R, the number of reinspected carcasses
    out of the n last potentially reinspected carcasses,
    for the given line. If emergency is True, gives
    R for the line's emergency r.i. entrance. Otherwise
    gives the line's normal r.i. entrance.
    """
    # Determine to-and-from Tx points
    assert not emergency, 'emergency line R not implemented yet'
    if line == 1:
        if emergency:
            fro, to = 5, 2
        else:
            fro, to = 14, 1
    else:
        if emergency:
            fro, to = 10, 8
        else:
            fro, to = 12, 7

    # Get order assignments
    order = bbh[bbh.Tx==fro].sort_index().reset_index()
    order = order[['uid','synthesised']]

    # TODO handle emergency entries, which may be multiple
    # matches for fro and to.
    def getord(uid):
        res = order.index[order.uid == uid]
        assert len(res) < 2, '{}: len={}, fro {}, to {}'.format(
                                uid, len(res), fro, to)
        if len(res) == 1:
            return res[0]
        return np.nan

    entering = bbh[bbh.Tx==to].sort_index()
    entering['pos'] = entering.uid.map(getord)

    # Function for computing the number of entrants
    # out of the last n potential entrants
    def f(p):
        return ((entering.pos <= p) & (entering.pos > p-n)).sum()

    res = entering.pos.map(f)
    res.name = 'R_{}{} N={}'.format(line, 'E' if emergency else '', n)

    if lag is not None:
        res = apply_lag(res, lag, update_name=True)

    return res

def sample_bag_of_alarms(sample_times, window, lag,
                         resolution='4 s', only_on_raise=True):
    boa = _bag_of_alarms(resolution=resolution,
                         only_on_raise=only_on_raise)

    # Drop data for days that won't be used
    boa = boa[np.in1d(boa.index.date, sample_times.date)]

    # Lag sample indices, and calculate window starts.
    # Retain order.
    laggedst = sample_times - pd.to_timedelta(lag)
    laggedstwindow = laggedst - pd.to_timedelta(window)
    slices = [slice(lstw, lst) for lst, lstw in zip(laggedst, laggedstwindow)]
    output = pd.DataFrame(0, index=sample_times, columns=boa.columns,
                          dtype=np.uint16)

    # Slow approach: Do the loop!
    for i, sl in tbiter.IProgressBar(enumerate(slices), len(slices)):
        # Get the sum of each window
        output.iloc[i] = boa[sl].sum(axis=0)
    return output

@tbfunc.persistent_lru_cache(2*24, ud.paths.Paths.cache)
@_ensure_alm
def _bag_of_alarms(alm, resolution='4 s', only_on_raise=True):
    if only_on_raise:
        almnr = alm.AlmNr[alm.AlmState == 1]
    else:
        almnr = alm.AlmNr

    df = pd.DataFrame(dtype=np.uint8)
    # Make bool column for each alarm number
    for v in almnr.unique():
        df['almnr_{}'.format(v)] = (almnr==v).astype(np.uint8).values

    df2 = pd.DataFrame(dtype=np.uint8)
    # Downsample to `resolution` steps, and merge duplicates
    # Need to do this column-wise because
    # memory otherwise becomes an issue
    for c in tbiter.IProgressBar(df.columns):
        df2[c] = df.pop(c).groupby(almnr.index.ceil(resolution)).sum()

    return df2

def apply_lag(ser, lag, update_name=True):
    """Lags ser so that the value that was at index T
    is known seen at T+lag (i.e. the new series is looking
    back in time)

    Modifies ser, and returns it.
    """
    if update_name:
        tbpd.append_to_name(ser, ' L={}'.format(lag))

    ser.index = ser.index + pd.to_timedelta(lag)
    return ser

def apply_summing_window(df, window, resolution, update_name=True):
    """Returns a resampled df, where each entry is the
    sum of values in the `window` time preceding the
    entry.
    """
    n = pd.to_timedelta(window).total_seconds() \
      / pd.to_timedelta(resolution).total_seconds()
    assert int(n) == np.round(n, 4), \
           '{} is not an integer; cannot use for rolling.'.format(n)

    df = df.resample(resolution).sum()
    df = df.fillna(0).rolling(int(n), 1).sum()

    if update_name:
        tbpd.append_to_name(df, ' W={}'.format(window))

    return df

def sample_line_specific(sample_times, lag, line,
                         delta_c_kwargs=None,
                         r_kwargs=None):
    """
    Returns samples for the given line at the given times,
    with independent variables from `lag` earlier than the
    dependent variable. The dependent variable is stored
    as column 'y'.

    delta_c_kwargs are keyword arguments for sample_delta_C.
    r_kwargs are keyword arguments for sample_R.
    """
    y = sample_C(sample_times, lag='0 s', line=line)
    y.name = 'y'
    c = sample_C(sample_times, lag=lag, line=line)
    dc = sample_delta_C(sample_times, lag=lag, line=line, **delta_c_kwargs)
    r = sample_R(sample_times, lag=lag, line=line, **r_kwargs)

    return pd.concat([y,c,dc,r], axis=1).fillna(0)

def sample_all(sample_times, lag,
               delta_c_kwargs=None,
               r_kwargs=None,
               bag_of_alarms_kwargs=None,):
    if delta_c_kwargs is None:
        delta_c_kwargs = {}
    if r_kwargs is None:
        r_kwargs = {}
    if bag_of_alarms_kwargs is None:
        bag_of_alarms_kwargs = {}

    LS = []
    for line in (1,2):
        LS.append(
            sample_line_specific(sample_times, lag=lag,
                line=line, delta_c_kwargs=delta_c_kwargs,
                r_kwargs=r_kwargs)
        )
    for i, df in enumerate(LS):
        df.columns = [x.replace('_1', '').replace('_2', '') for x in df.columns]
        df['line'] = i+1
    boa = sample_bag_of_alarms(sample_times, lag=lag, **bag_of_alarms_kwargs)

    LS = (pd.concat([L, boa], axis=1) for L in LS)
    return pd.concat(LS, axis=0).sort_index()
