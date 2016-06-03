import itertools
import multiprocessing

from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import seaborn as sns

from . import basics
from . import txgraph

import tbtools.iter as tbiter


def plot(bbh, ax=None, color='g', spread='auto', alpha='auto',
         labels=None, legend=True,
         end_marker=None, end_marker_color=None,
         **basicplot_kwargs):
    """Draws a flow map where each carcass path is represented
    by a thin, transparent line.

    Returns the used figure axis.

    bbh is a BitBusHist dataframe or a groupby object from one.
    ax is a plot axis to draw on.
    color, spread, alpha, and labels can be dicts mapping
        bbh groupby keys to values, in case bbh is a groupby
        object.
    color is a matplotlib color (string) for the path color.
    spread is 'auto' or a float. It is used for spreading the
        carcass paths around their Tx points.
    alpha is the opacity of the path lines.
    labels are used if bbh is a groupby object; it is a dict
        mapping keys to legend labels.
    legend is bool. Set to False to drop the legend for groupbys.
    end_marker is None or a matplotlib marker to mark path
        endings
    end_marker_color is the color for the end marker.
    """
    if isinstance(bbh, pd.core.groupby.DataFrameGroupBy):
        return plot_by_hue(bbh, ax, color, spread, alpha, labels, legend,
                           **basicplot_kwargs)

    paths = _bbh_to_paths(bbh)
    if spread == 'auto':
        # Pure magic. Found by what I though looked best.
        val = (np.log(len(paths))/np.log(1.5))/60
        spread = np.clip(val, 0.05, .25)
    paths = _add_circular_noise(paths, spread)
    segments = _paths_to_segments(paths)

    if ax is None:
        bpk = {
            'tx': True,
            'descriptions': False,
            'edges': False,
        }
        bpk.update(basicplot_kwargs)
        ax = basics.plot(**bpk)

    if alpha == 'auto':
        # Pure magic. Found by what I though looked best.
        # Log_1.5
        val = 1/(np.log(len(paths))/np.log(1.5))
        alpha = np.clip(val, 0.01, 1.0)

    _draw_segments(ax, segments, color, alpha)
    if end_marker is not None:
        _draw_markers(ax, segments, end_marker_color or color, end_marker)

    return ax

from collections import defaultdict
def _val_to_dict(val):
    if type(val) == dict:
        return val
    return defaultdict(lambda: val)

def plot_by_hue(bbhgroups, ax, colors, spread, alpha, labels, legend,
                **basicplot_kwargs):
    if not isinstance(colors, dict):
        colors = {k:v for (k,_),v in zip(
                    bbhgroups,
                    sns.color_palette())}
    labels = _val_to_dict(labels)
    spread = _val_to_dict(spread)
    alpha = _val_to_dict(alpha)

    legend_patches = []
    for key, gr in bbhgroups:
        label = labels[key] or key
        color = colors[key]

        ax = plot(gr, ax=ax, color=color,
                  spread=spread[key], alpha=alpha[key],
                  **basicplot_kwargs)
        legend_patches.append(
            sns.mpl.patches.Patch(color=color,
                                  label=label))
    # Plot the legend
    if legend:
        sns.plt.legend(handles=legend_patches)
    return ax

def _bbh_to_paths(bbh):
    groupby = 'uid' if 'uid' in bbh else 'HangerID'
    with multiprocessing.Pool() as p:
        paths = p.map(_uid_group_to_paths_worker, bbh.Tx.groupby(bbh[groupby]))
    return paths

def _uid_group_to_paths_worker(uid_txs):
    uid, txs = uid_txs
    return np.array(list(map(txgraph.positions.get, txs)), dtype=float)

def _add_circular_noise_worker(path_noisescale):
    path, spread = path_noisescale
    return path + _circular_noise(len(path), spread)

def _add_circular_noise(paths, spread):
    paths = list(map(_add_circular_noise_worker,
                     zip(paths, itertools.repeat(spread))))
    # This gives bad 'random' samples:
    # with multiprocessing.Pool() as p:
    #     paths = p.map(_add_circular_noise_worker,
    #                   zip(paths, itertools.repeat(spread)))
    return paths

def _circular_noise(length, scale):
    n = np.random.uniform(-scale, scale, (length, 2))
    lengthssqr = n.sum(axis=1)**2
    fix = lengthssqr > scale**2

    deadman = 6
    curscale = scale
    while any(fix):
        deadman -= 1
        if deadman < 0:
            return n
        n[fix] = np.random.uniform(-curscale, curscale, (sum(fix), 2))
        lengthssqr = n.sum(axis=1)**2
        fix = lengthssqr > scale**2
        curscale *= 0.9

    return n

def _paths_to_segments(paths):
    with multiprocessing.Pool() as p:
        res = p.map(_path_to_segment_worker, paths)
    return res

def _path_to_segment_worker(path):
    """path is an array or list of (x,y)
    Returns an np.array of
        [[x0, y0],
         [x1, y1]]
    """
    pair = list(tbiter.pairwise(path))
    arr = np.array(pair)
    return arr

def _draw_segments(ax, segments, color, alpha):
    for seg in segments:
        lc = LineCollection(seg, linewidths=1, color=color, alpha=alpha)
        ax.add_collection(lc)
    return ax

def _draw_markers(ax, segments, color, end_marker):
    coords = list(zip(*[seg[-1][-1] for seg in segments]))
    ax.scatter(coords[0], coords[1], color=color, marker=end_marker, zorder=4)
    return ax