import tbtools.iter as tbiter

from . import basics

def plot(bbh, count_visits=True, show_descriptions=True,
         color='#FF0000', alpha=.5, ax=None,
         min_count=2, **basicsplot_kwargs):
    if 'uid' in bbh:
        groupby = 'uid'
    else:
        groupby = 'HangerID'

    if ax is None:
        ax = basics.plot(descriptions=show_descriptions, **basicsplot_kwargs)

    for key, vals in bbh.groupby(groupby):
        for s,t in tbiter.pairwise(vals.Tx):
            basics.arrow(ax, s, t, color=color, alpha=alpha,
                         zorder=1.5)

    if count_visits:
        d = {}
        for tx in bbh.Tx.unique():
            count = int((bbh.Tx==tx).sum())
            if count >= min_count:
                d[tx] = count
        basics.text(ax, d, color='g',
                    xmod=-.15)

    return ax
