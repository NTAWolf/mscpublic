import pandas as pd
import numpy as np

import seaborn as sns


#
# Plot types
#

def categorical_value_differences(a, b, figsize=(14,5), ylim=None):
    """For two series with the same categorical index, and
    subtractable values, shows a plot where the distance between
    each value pair is a (red) vertical line.
    """
    labels, xpos = a.index, range(len(a.index))
    sns.plt.figure(figsize=figsize)

    sns.plt.bar(left=xpos,
                height=(a-b).values,
                width=0.1,
                bottom=b.values,
                align='center',
                orientation='vertical',
                color='r')
    sns.plt.xticks(xpos, labels, rotation='vertical')
    sns.plt.xlim((-1, len(labels)))
    if ylim is not None:
        sns.plt.ylim(ylim)

    sns.plt.scatter(x=xpos, y=b.values, color='k')



#
# Heatmap for determining how alike two binary dataframes are
# Used for alarm comparison
#

def monocolor(r,g,b):
    """Returns a matplotlib Colormap with only the given color.
    """
    cdict = {}
    for key, v in zip(('red', 'green', 'blue'), (r,g,b)):
        cdict[key] = ((0,v,v), (1,v,v))

    cmap = sns.mpl.colors.LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

def duocolor(rgb1, rgb2):
    """Returns a matplotlib Colormap going from rgb1 to rgb2.
    """
    cdict = {}
    for key, v1, v2 in zip(('red', 'green', 'blue'), rgb1, rgb2):
        cdict[key] = ((0,v1,v1), (1,v2,v2))

    cmap = sns.mpl.colors.LinearSegmentedColormap('custom_cmap', cdict)
    return cmap


def heatmap_discrete_levels(data, key2color, fill_color=(1,1,1), **heatmap_kwargs):
    """data can be a DataFrame (with multiple value levels), or
    a tuple, list, or dict of DataFrames with True-False-like values.

    In the former case, key2color[key] is the color for data[data==key].
    In the latter case, key2color[key] is the color for data[key].

    key2color is a dict. Values must be r,g,b tuples [0,1]
    """
    filled = False
    for key, rgb in key2color.items():
        if isinstance(data, pd.DataFrame):
            vals = (data == key)
        else:
            vals = data[key]

        if filled:
            if not vals.any().any():
                # We will get an error if we try to
                # render this.
                continue
            color = monocolor(*rgb)
            vals[~vals.astype('bool')] = np.nan
            vals = vals.astype('float16')
        else:
            color = duocolor(fill_color, rgb)
            filled = True

        kwargs = {'cbar':False}
        kwargs.update(heatmap_kwargs)
        sns.heatmap(vals, cmap=color, **kwargs)

def dummy_variable_overlaps(a, b, name_a='a', name_b='b',
                            figsize=(10,None),
                            drop_empty_rows=True, drop_empty_cols=True,
                            x_label='', y_label='',
                            force_show_all_labels=False,
                            datetimeindex_as_time=True):
    """a and b are pandas DataFrames with same index and columns.
    All values are expected to be 0 or 1.

    figsize can have an element that is None. The correct size of the
        other dimension will then be calculated herein.

    Draws a matrix showing where they have overlapping 1s (white),
    where a=1 and b=0 (red), a=0 and b=1 (blue), and both are 0 (black).

    Beware that this takes O(n^2) time:

        plottingtime = pd.DataFrame(
                            {'nrows':[20,  40,   60,  160,  460,
                                      1060, 1500, 2060, 2500],
                             'time': [345, 442, 558, 1500, 5940,
                                      21300, 42100, 65000, 101000]})
        sns.lmplot(x="nrows", y="time", data=plottingtime,
                     order=2, ci=None, scatter_kws={"s": 80});
    """
    if drop_empty_cols:
        something = (a | b)
        keep = a.columns[something.sum(axis=0) > 0]
        a, b = a[keep], b[keep]
    if drop_empty_rows:
        something = (a | b)
        keep = a.index[something.sum(axis=1) > 0]
        a, b = a.loc[keep], b.loc[keep]

    both = a&b
    only_a = a&(~b)
    only_b = (~a)&b

    # Figure out what size we should draw it in
    w_to_h = a.shape[0] / a.shape[1]
    if figsize[0] is None:
        figsize = (figsize[1] / w_to_h, figsize[1])
    elif figsize[1] is None:
        figsize = (figsize[0], figsize[0] * w_to_h)

    sns.plt.figure(figsize=figsize)
    rows_per_inch = a.shape[0] / figsize[1]

    LABELS_PER_INCH = 2.5
    xticklabels = int(a.shape[1] / (figsize[0]*LABELS_PER_INCH))
    yticklabels = int(a.shape[0] / (figsize[1]*LABELS_PER_INCH))


    pwargs = {'cbar':False, 'square':True, 'linewidth':0.2}

    if isinstance(a.index, pd.DatetimeIndex) and datetimeindex_as_time:
        c = max(yticklabels, 1) if not force_show_all_labels else 1
        pwargs['yticklabels'] = ['{:%H:%M:%S}'.format(d) if i%c == 0 else ''
                                    for i,d in enumerate(a.index)]

    # Make nicely spaced labels.
    if not force_show_all_labels:
        if xticklabels > 1: pwargs['xticklabels'] = xticklabels
        if yticklabels > 1 and not 'yticklabels' in pwargs:
            pwargs['yticklabels'] = yticklabels

    if rows_per_inch < 8:
        pwargs['linewidths'] = 0.05
        pwargs['linecolor'] = (.2, .2, .2)

    # pwargs['linewidths'] = 0.1

    colors = {0:(0,0,0), 1:(1,0,0), 2:(0,1,1)}
    heatmap_discrete_levels((both, only_a, only_b), colors, **pwargs)

    # Write what the colors symbolize.
    legend_y_gap = rows_per_inch / 1.5
    legend_x_gap = 2.75
    legend_x = a.shape[1] + 0.5
    legend_y = a.shape[0] // 2 + rows_per_inch
    legend_text = ('{a} & {b}', '{a} only', '{b} only')
    legend_text = map(lambda x: x.format(a=name_a, b=name_b), legend_text)
    sns.plt.axvline(legend_x - 1., color=(.9,.9,.9))
    # sns.plt.axvline(legend_x - 1., 0.45, 0.625, color='grey')
    for i, txt in enumerate(legend_text):
        x = legend_x
        y = legend_y - (i*legend_y_gap)
        c = colors[i]
        sns.plt.text(x,y, '█', color=c)
        sns.plt.text(x+legend_x_gap, y, txt)

    sns.plt.xlabel(x_label)
    sns.plt.ylabel(y_label)