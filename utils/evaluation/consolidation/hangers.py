import pandas as pd
import seaborn as sns

from ... import data as ud
from ... import plotting as up


def _day_from_Timestamp(pc):
    return '{:%Y-%M-%D}'.format(pc.Timestamp.min())


def _day_from_cons(mc):
    return '{:%Y-%M-%D}'.format(pc.Timestamp_tx14.min())


def _get_whatever_is_missing(day, precons_dirty, precons_clean,
                             cons_report, cons_final, bbh):
    """Given either day or one of the remaining arguments, infers
    what is needed, gets it, and returns it in the same order
    as the args.
    """
    if day is None:
        if precons_dirty is not None:
            day = _day_from_Timestamp(precons_dirty)
        elif precons_clean is not None:
            day = _day_from_Timestamp(precons_clean)
        elif cons_report is not None:
            day = _day_from_precons(cons_report)
        elif cons_final is not None:
            day = _day_from_precons(cons_final)
        elif bbh is not None:
            day = _day_from_Timestamp(bbh)
        else:
            raise ArgumentError('Not enough non-None arguments.')

    if cons_final is None:
        cons_final = ud.consolidation.get(day)
    if cons_report is None:
        cons_report = ud.consolidation.get(day, settings='report')
    if precons_dirty is None:
        precons_dirty = ud.preconsolidated.get(day)
    if precons_clean is None:
        precons_clean = ud.preconsolidated.clean(precons_dirty)
    if bbh is None:
        bbh = ud.raw.get('bbh', day)

    return day, precons_dirty, precons_clean, cons_report, cons_final, bbh


def plot_shares(day=None, precons_dirty=None, precons_clean=None,
                cons_report=None, cons_final=None,
                bbh=None, resample_interval='10 min', figsize=(12,8)):
    """Plot HangerID shares of the total possible number of hangerids for
    each of the datasets: preconsolidated (not cleaned), cleaned
    preconsolidated, consolidated report-style, and final consolidation.
    """
    day, precons_dirty, precons_clean, cons_report, cons_final, bbh =\
        _get_whatever_is_missing(day, precons_dirty, precons_clean,
                                 cons_report, cons_final, bbh)

    bbh = bbh[bbh.Tx == 14]
    bbh_count = bbh.HangerID.resample(resample_interval).count()

    precons_dirty = precons_dirty.set_index('Timestamp')
    data = {
        'cons final': cons_final,
        'cons report': cons_report,
        'precons cleaned': precons_clean,
        'precons original': precons_dirty,
    }

    for k in data:
        data[k] = bbh.HangerID.isin(data[k].HangerID).resample(
            resample_interval).sum() / bbh_count

    df = pd.DataFrame(data, index=bbh_count.index)
    with sns.color_palette("colorblind"):
        with sns.axes_style("whitegrid"):
            plt = df.plot(figsize=figsize)

    sns.plt.title('HangerIDs from raw data seen in consolidated data')
    sns.plt.ylabel(
        'Fraction of HangerIDs of total possible, excluding ghost hanger')
    sns.plt.xlabel(
        'Timestamp on {}. Bins of {}.'.format(day, resample_interval))
    up.lim_expand(plt, 1.1, relative=True)
