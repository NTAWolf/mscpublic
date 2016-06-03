import numpy as np

from matplotlib.collections import LineCollection

from . import basics
from . import txgraph

def plot(tx_count_dict, max_width=.4, height=14,
        textcolor='k',
        bars=True, barcolor='#55FF55', baroffset=.1,
        position='right', ax=None, **basicsplot_kwargs):
    """
    max_width is the max width of the bars, in data units.
    height is the height of the bars, in pixels
    position can be 'right' or 'left' of the Tx points.
    """
    if ax is None:
        ax = basics.plot(**basicsplot_kwargs)

    xmod = .3
    if position == 'right':
        horizontalalignment = 'left'
    if position == 'left':
        xmod = -xmod
        horizontalalignment = 'right'
    basics.text(ax, tx_count_dict, format='{:,}',
                color=textcolor, xmod=xmod, zorder=5)

    widths = np.array(list(tx_count_dict.values()), dtype=float)
    # Normalize so the widest bar will be max_width
    widths *= (max_width/np.max(widths))

    xs, ys = list(zip(*[txgraph.positions[tx] for tx in tx_count_dict]))

    # Bars
    if bars:
        if position == 'right':
            xs = [x+baroffset for x in xs]
        else:
            xs = [x-baroffset for x in xs]
            widths *= -1
        horizontal_rects(ax, xs, ys, widths, height, color=barcolor, zorder=4)

    return ax

def horizontal_rects(ax, xs, ys, widths, heights, **LineCollection_kwargs):
    """Draw 'rectangles' as cheap lines.
    ax is a matplotlib axis.
    xs is a listlike of x values in data units - one of
        the rect endpoints
    ys is a listlike of y values in data units - which is
        the same for both ends of the rect
    widths is a listlike of the desired width of the rects
        (in data units)
    heights is a number or a listlike. It is in pixels.
    """
    if isinstance(heights, float) or isinstance(heights, int):
        heights = np.repeat(heights, len(xs))

    segments = np.array([ ((x,y), (x+w, y)) for x,y,w in zip(xs,ys,widths)])

    lc = LineCollection(segments, linewidths=heights, **LineCollection_kwargs)
    ax.add_collection(lc)

    return ax
