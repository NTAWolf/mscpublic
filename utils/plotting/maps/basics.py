import seaborn as sns
import numpy as np

from . import txgraph

PLT_TEXTHEIGHT = 0.2

def plot(tx=True, descriptions=False, edges=True,
         subset=None, figsize=(10,12), height=None,
         title=None, edge_color=None, tx_markersize=None):
    """Draw a plot of the basic map. Arguments are
    booleans, indicating whether tx points should be drawn,
    and whether edges should be drawn.

    tx point are drawn in zorder 2
    edges are drawn in zorder 1
    """
    if height is not None:
        # Preserve 10,12 aspect ratio
        figsize = ((5/6)*height, height)

    with sns.axes_style("white"):
        fig, ax = sns.plt.subplots(figsize=figsize)
        if edges:
            _edges(ax, color=edge_color)
        if tx:
            _tx_points(ax, descriptions=descriptions,
                       markersize=tx_markersize)

    adjust_lims(ax, subset)
    clear_ticks(ax)

    if title is not None:
        ax.set_title(title)

    return ax

def edges_from_subset(subset):
    """Returns left, right, bottom, top
    """
    txpos = _dict_subset(subset, txgraph.positions).values()
    xs,ys = zip(*txpos)
    return min(xs), max(xs), min(ys), max(ys)

def adjust_lims(ax, subset=None,
                topoffset=1, bottomoffset=.5,
                leftoffset=.5, rightoffset=.5):
    left, right, bottom, top = edges_from_subset(subset)
    ax.set_xlim((left - leftoffset, right + rightoffset))
    ax.set_ylim((bottom - bottomoffset, top + topoffset))

def clear_ticks(ax):
    ax.tick_params(axis='both', which='both',
                   bottom='off', top='off', labelbottom='off',
                   left='off', right='off', labelleft='off')

def _dict_subset(keysubset, dictlike):
    if keysubset is None:
        return dictlike
    return {k:dictlike[k] for k in keysubset}

def _tx_points(ax, markers=True, numbers=True, descriptions=True,
              subset=None, zorder=2, markersize=None):
    if markers:
        _tx_markers(ax, subset=subset, zorder=zorder, markersize=markersize)
    if numbers:
        _tx_numbers(ax, subset=subset, zorder=zorder)
    if descriptions:
        _tx_descriptions(ax, subset=subset, zorder=zorder)
    return ax

def _tx_markers(ax, color=None, markersize=None,
               subset=None, **mpl_arrow_kwargs):
        """Draw tx markers.
        markersize defaults to 15
        color defaults to '$5555FF'
        """
        markersize = markersize or 15
        color = color or '#5555FF'

        txpos = _dict_subset(subset, txgraph.positions)
        xs,ys = zip(*txpos.values())
        ax.plot(xs, ys, 'o', color=color, markersize=markersize,
                **mpl_arrow_kwargs)

        return ax

def _tx_descriptions(ax, color='k', voffset=0.3, subset=None,
                    **mpl_text_kwargs):
    txpos = _dict_subset(subset, txgraph.positions)
    desc = _dict_subset(subset, txgraph.descriptions)

    text(ax, desc, color=color, ymod=voffset, **mpl_text_kwargs)

def _tx_numbers(ax, color='w', subset=None, **mpl_text_kwargs):
    txpos = _dict_subset(subset, txgraph.positions)

    nums = {tx:str(tx) for tx in txpos}

    text(ax, nums, color=color, **mpl_text_kwargs)

def arrow(ax, source, target,
          color='#FF0000', alpha=.5,
          length_reduce_constant=0.15,
          width=.025, head_width=.15, head_length=.25,
          **mpl_arrow_kwargs):

    x, y = txgraph.positions[source]
    target = txgraph.positions[target]
    dx, dy = target[0]-x, target[1]-y

    length = np.sqrt(dx**2 + dy**2)
    ratio = (length - length_reduce_constant) / length
    dx *= ratio
    dy *= ratio

    ax.arrow(x, y, dx, dy,
             fc=color, ec=color, alpha=alpha,
             width=width,
             head_width=head_width, head_length=head_length,
             length_includes_head=True, **mpl_arrow_kwargs)

    return ax

def _edges(ax, color=None, subset=None, zorder=1):
    """Draw directed edges as arrows.
    color defaults to '#DDEEEE'
    """
    color = color or '#DDEEEE'
    txedge = txgraph.edges_list
    if subset is not None:
        # TODO decide on how to handle edge subsetting
        pass

    for source, target in txedge:
        arrow(ax, source=source, target=target,
              color=color, zorder=zorder)

    return ax


def _numeric_to_addition_func(val):
    if isinstance(val, int) or isinstance(val,float):
        return lambda x: val+x
    return val

def text(ax, tx_to_text_dict, color='k', format='{}',
          xmod=0, ymod=0, xymod=None,
          horizontalalignment='center',
          **mpl_kwargs):
    """Writes text on the axis. tx_to_text_dict is a dict.
    xmod and ymod can be numeric values to be added to the coordinates,
    or functions that take their existing respective values and returns
    a modified version of them.
    xymod is None or a function that takes x,y and returns x,y. It is
    applied after xmod and ymod.
    """
    xmod = _numeric_to_addition_func(xmod)
    ymod = _numeric_to_addition_func(ymod)

    for k,v in tx_to_text_dict.items():
        x,y = txgraph.positions[k]
        x = xmod(x)
        y = ymod(y)
        if xymod is not None:
            x,y = xymod(x,y)
        ax.text(x, y, format.format(v), color=color,
               horizontalalignment=horizontalalignment,
               verticalalignment='center',
               clip_on=True,
               **mpl_kwargs)
