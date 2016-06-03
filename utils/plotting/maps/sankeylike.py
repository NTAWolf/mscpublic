from collections import Counter

from matplotlib.collections import LineCollection
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

from . import txgraph


def fill_edges(bbh, begin=None, end=None):
    assert 'uid' in bbh, "Use enhanced bitbushist"
    if begin is not None:
        bbh = bbh[begin:end]

    edges = Counter()
    for uid, tx in bbh.Tx.groupby(bbh.uid):
        tx = tx.sort_index().values
        edges.update(zip(tx[:-1], tx[1:]))

    real_edges = {k: 0 for k in txgraph.edges_list}
    false_edges = {}
    for k, v in edges.items():
        if k in real_edges:
            real_edges[k] = v
        else:
            false_edges[k] = v

    return real_edges, false_edges


def get_segments_and_counts(edge_counts):
    """edge_counts is a dict with k,v pairs like
        (1,13): 242
    """
    seg = txgraph.edges_to_line_segments(edge_counts)
    counts = np.array([x for x in edge_counts.values()])
    return seg, counts

# def annotate_visitor_count(edge_counts):


def draw_segments(*seg_width_color, figsize=(10, 10)):
    """seg_width_color is any number of tuples consisting of
        segments: np.array shape N,2,2, inner vals being
            [[start.x, start.y],
             [  end.x,   end.y]]
        width: list or array length N
        color: a single matplotlib color specification
    """
    with sns.axes_style("white"):
        fig, a = plt.subplots(figsize=figsize)
        for seg, width, color in seg_width_color:
            lc = LineCollection(seg, linewidths=width, color=color)
            a.add_collection(lc)

    a.set_xlim(-3, 3)
    a.set_ylim(-1, 11)
    # a.autoscale_view()
    plt.tick_params(
        axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(
        axis='y', which='both', left='off', right='off', labelleft='off')


def draw(bbh):
    """bbh is enhanced with uids column
    """
    r, f = fill_edges(bbh)
    # max_density = max(max(r.values()), max(f.values()))
    rs, rc = get_segments_and_counts(r)
    fs, fc = get_segments_and_counts(f)

    # One line for each time path was taken
    r = np.repeat(rs, rc, axis=0)
    f = np.repeat(fs, fc, axis=0)

    # Add noise for nice spread
    hit_area_scale = 0.15 # stay within .15 units
    r += hit_area_scale*np.random.uniform(-1, 1, r.shape)
    f += hit_area_scale*np.random.uniform(-1, 1, f.shape)

    n_traces = len(bbh.uid.unique())
    alpha_r = max(1e-2, 0.97**(n_traces/2))  # 1/np.log(max_density)
    alpha_f = min(1, 2*alpha_r)

    # make lines that are in smaller groups wider
    rc2 = np.min([np.repeat(4, len(rc)), np.log(max(rc) - rc + 1)], axis=0)
    fc2 = np.min([np.repeat(4, len(fc)), np.log(max(fc) - fc + 1)], axis=0)

    # One line for each time path was taken
    rc = np.repeat(rc2, rc, axis=0)
    fc = np.repeat(fc2, fc, axis=0)

    draw_segments((r, [1 for _ in range(len(r))],
                   (0, 0, 1, alpha_r)),
                  (f, [1 for _ in range(len(f))],
                   (1, 0, 0, alpha_f)))
