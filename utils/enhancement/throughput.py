"""This module contains methods for calculating and plotting the number of registered
hangers passing by one or more specific Tx points for a given period, at a given
bin size (frequency)
"""

import pandas as pd
import numpy as np

import seaborn as sns

import utils.data as ud
import utils.plotting as up

def _select_indexrange(indexed, a=None, b=None):
    if b is None:
        return indexed[a]
    if a is not None:
        return indexed[a:b]
    return indexed

def _force_iterable(val):
    if not _is_iterable(val):
        val = (val,)

    return val

def _is_iterable(val):
    return (isinstance(val, list) or isinstance(val, tuple))

def _contains_iterable(val):
    if not _is_iterable(val):
        return False
    if any([_is_iterable(v) for v in val]):
        return True
    return False


def plot_throughput(tx, period, period_end=None, frequency='5 min', figsize=(15,7)):
    """Plot the rate at which non-ghost hangers are passing through the given tx
    point(s). You can give a single integer as a tx, or a list or tuple for a grouped
    view, or a list/tuple containing one or more sub-list/tuples for comparing
    groups of txs.

    period and period_end: strings like '2012-09-28 15:00:00'
        period_end can be None, in which case what is encompassed by period
        is selected.
    frequency: string like '10 min', '1'
    """

    if _contains_iterable(tx):
        fig = _plot_throughput_overlap(tx, period, period_end,
                                 frequency, figsize)
    else:
        throughput = get_throughput(tx, period, period_end, frequency)
        throughput.plot(figsize=figsize)

    _set_plot_title_and_labels(fig, tx, period, frequency)

def _plot_throughput_overlap(tx_groups, period, period_end,
                            frequency, figsize):
    tps = [get_throughput(tx, period, period_end, frequency) for tx in tx_groups]
    df = pd.concat(tps, axis=1, keys=['Tx {}'.format(tx) for tx in tx_groups])
    fig = df.plot(figsize=figsize)
    return fig

def _set_plot_title_and_labels(fig, tx, period, frequency):
    tx = _force_iterable(tx)

    if _contains_iterable(tx):
        title = 'Comparison of throughputs on {}'.format(', '.join(['Tx {}'.format(t) for t in tx]))
    else:
        desc = list(map(lambda x: up.mapping.NN_map.get_by_tx(x).description, tx))
        title = 'Throughput at {txs} on {period}'.format(
            txs=', '.join(['Tx {} ({})'.format(tx, d) for tx,d in zip(tx, desc)]),
            period=period
        )

    sns.plt.title(title)
    sns.plt.ylabel('Number of carcasses per {}'.format(frequency))
    up.lim_expand(fig, 1.1, relative=True)
    sns.plt.tight_layout()

def get_throughput(tx, period=None, period_end=None, frequency='5 min'):
    tx = _force_iterable(tx)

    bbh = ud.raw.get('bitbushist', period, period_end)
    bbh = bbh[bbh.Tx.isin(tx)]

    return bbh.resample(frequency).count().Tx