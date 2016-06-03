import matplotlib as mpl
import numpy as np
import seaborn as sns

LABELS_PER_INCH = 2.5
# In RegularPolyCollection, at least
POINTS_PER_INCH = 80

def plot_poly(numsides, coords, sizes, colors='k', figsize=None, ax=None,
              lims=None, **rpc_kwargs):
    """numsides is the number of sides of the polygons
    coords is a 2d structure, where the second dimension is x and y
        coordinates.
    sizes is a 1d structure, with the size of each polygon given as
        the radius of a superscribed circle.
        Can also be a single value for all coords.
    colors is either a listlike or a single matplotlib color.
    ax is an existing axis. If none is given, this method creates one.

    rpc_kwargs:
        rotation is in radians.

    Returns ax

    Example:
        _ = plot_poly(3, [[0.5,2], [1,1.8], [2,0.2]], 10, colors='g')
    """
    sizes = np.pi * np.square(sizes)
    if isinstance(sizes, float) or isinstance(sizes, int):
        sizes = np.repeat(sizes, len(coords))

    with sns.axes_style("white"):
        if ax is None:
            _, ax = mpl.pylab.subplots(figsize=figsize)

        rpc = mpl.collections.RegularPolyCollection(
                    numsides=numsides,
                    sizes=sizes,
                    offsets=coords,
                    transOffset=ax.transData,
                    **rpc_kwargs)
        trans = mpl.transforms.Affine2D().scale(ax.figure.dpi / 72.0)
        rpc.set_transform(trans)  # the points to pixels transform
        ax.add_collection(rpc, autolim=True)
        rpc.set_color(colors)
        if lims is None:
            ax.autoscale_view()
        else:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])

    return ax

def plot_squares(coords, sizes, **plot_poly_kwargs):
    """coords is a 2d structure, where the second dimension is x and y
        coordinates.
    sizes is a 1d structure, with size of each square given in side length.
        Can also be a single value for all coords.

    plot_poly_kwargs:
        rotation is in radians. Defaults to np.pi/4, namely upright squares.
        colors is either a listlike or a single matplotlib color.
        ax is an existing axis. If none is given, this method creates one.

    Returns ax

    Example:
        ax = plot_squares([[1,2], [2,2], [3,-1]], [10, 10, 50])
        _ = plot_squares([[0,1], [1,3], [3,1]], [20, 20, 30], colors='r', ax=ax)
        _ = plot_squares([[0,2], [1,1], [2,0]], 10, colors='g', ax=ax)
    """
    sizes = np.sqrt(np.square(sizes) / 2)
    if not 'rotation' in plot_poly_kwargs:
        plot_poly_kwargs['rotation'] = np.pi/4
    return plot_poly(numsides=4, coords=coords, sizes=sizes, **plot_poly_kwargs)

def infer_settings(rows=None, cols=None, unitsize=None, gap=0, figsize=None,
                   aspect=None):
    """Needs at least
        ((rows & cols) | (aspect & cols)) and
        (figsize[0] | figsize[1] | unitsize)

    unitsize is in points.
    gap is in points, but is only rough.
    figsize is a tuple or None.

    returns unitsize, figsize, x_every_n_label, y_every_n_label
    """
    w_to_h = aspect or rows/cols

    # Determine figure size
    if figsize is None:
        # I don't understand this, but it has to be done:
        gap = unitsize + gap
        width = cols*unitsize + (cols-1)*gap
        width /= POINTS_PER_INCH
        figsize = (width, None)
    if figsize[0] is None:
        figsize = (figsize[1] / w_to_h, figsize[1])
    elif figsize[1] is None:
        figsize = (figsize[0], figsize[0] * w_to_h)

    # Determine square size
    if unitsize is None:
        unitsize = (POINTS_PER_INCH * figsize[0]/rows) - gap/2

    x_every_n_label = int(cols / (figsize[0]*LABELS_PER_INCH))
    y_every_n_label = int(rows / (figsize[1]*LABELS_PER_INCH))

    return unitsize, figsize, x_every_n_label, y_every_n_label

def bilinear(interp, *args):
    if not isinstance(interp, list) and not isinstance(interp, tuple):
        interp = [interp]
    output = []
    for v, (mi, ma) in zip(interp, args):
        output.append(v*(ma-mi) + mi)
    return output

def custom_legend(ax, position, colors, labels):
    """Position in normalized plot coordinates
    """
    pos = bilinear(position, ax.get_xlim(), ax.get_ylim())
    print('ax.figure.get_size_inches()',ax.figure.get_size_inches())
    print('pos',pos)
    labheight = 1/LABELS_PER_INCH
    y_start = pos[1] + labheight*(len(labels)/2)
    legend_x_gap = bilinear(0.2, ax.get_xlim())[0]

    for i, (color, label) in enumerate(zip(colors, labels)):
        x = pos[0] + legend_x_gap
        y = y_start - (i*labheight)
        ax.text(x,y, '█', color=color)
        ax.text(x+2*legend_x_gap, y, label)
