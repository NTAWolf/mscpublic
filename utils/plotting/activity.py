import numpy as np

from . import primitives as prim

def plot(df, color='k', ax=None, size=None, gap=3, figsize=(10,None),
         labels=True, force_show_all_labels=False,
         **plot_squares_kwargs):
    """
    Example:
        df = pd.DataFrame(
            {'a':[1,0,1,0,1,0], 'b':[0,0,1,1,0,1], 'c':[1,0,0,0,0,0]},
            index=list('654321'))
        ax = plot_df_activity(df.astype(bool), 'g')
        ax = plot_df_activity(df-1, 'r', ax=ax)
    """
    labx = df.columns
    laby = df.index
    coords = np.array(np.where(df.T.values)).T

    size, figsize, x_every_n_label, y_every_n_label = \
        prim.infer_settings(*df.shape, size, gap, figsize)

    ax = prim.plot_squares(coords, colors=color, ax=ax,
                      sizes=size, figsize=figsize,
                      **plot_squares_kwargs)

    if labels:
        # Handle labelling
        if not force_show_all_labels:
            x_every_n_label = max(1, x_every_n_label)
            y_every_n_label = max(1, y_every_n_label)
            labx = [L if (i%x_every_n_label)==0 else '' for i,L in enumerate(labx)]
            laby = [L if (i%y_every_n_label)==0 else '' for i,L in enumerate(laby)]

        ax.set_xticks(range(len(labx)))
        ax.set_xticklabels(labx, rotation='vertical')

        ax.set_yticks(range(len(laby)))
        ax.set_yticklabels(laby)

    return ax

def plot_all(dfs, colors, legend=None, **plot_kwargs):
    put_labels = True
    ax = None
    for df, col in zip(dfs, colors):
        ax = plot(df, color=col, labels=put_labels, ax=ax, **plot_kwargs)
        put_labels = False
    if legend is not None:
        prim.custom_legend(ax, (1,0.5), colors, legend)

def compare(dfa, dfb, name_a='a', name_b='b', **plot_kwargs):
    dfa = dfa.astype(bool)
    dfb = dfb.astype(bool)
    vals = (dfa&dfb, dfa&~dfb, dfb&~dfa)
    colors = ('k', 'r', (0,1,1))
    labels = ('{} & {}'.format(name_a, name_b),
              '{} only'.format(name_a),
              '{} only'.format(name_b),)
    plot_all(vals, colors, labels, **plot_kwargs)
