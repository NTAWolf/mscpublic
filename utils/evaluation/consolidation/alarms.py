import pandas as pd

from ...plotting import categorical as plotcat

def get_alarm_columns(df):
    return [x for x in df.columns if x[0] == 'E' and x[1:].isdigit()]

def alarms_only(*args):
    if len(args) == 1:
        df = args[0]
        return df[get_alarm_columns(df)]
    return  [alarms_only(x) for x in args]

def get_aligned_alarms(pc, mc):
    """Lines preconsolidated and my consolidated data up
    so that the alarms for each hangerid and intestine remover
    timestamp are in the same index and column.

    pc is the preconsolidated data.
    mc is my own consolidation.
    """

    pckeys = ['HangerID', 'OrganTimestamp']
    mckeys = ['HangerID', 'Timestamp_organ']

    mix = pd.merge(pc[pckeys], mc[mckeys],
                   left_on='HangerID', right_on='HangerID')

    # Get rid of merge errors where the timestamps are more different than
    # the rounding errors.
    keep = abs((mix.OrganTimestamp - mix.Timestamp_organ).astype('timedelta64[s]')) <= 1
    mix = mix[keep]

    pc_alm = get_alarm_columns(pc)
    mc_alm = get_alarm_columns(mc)

    # Put the Timestamp_organ column into the corresponding spots in pc.
    pc = pc.set_index('OrganTimestamp')
    pc['mc_timestamp_organ'] = mix.set_index('OrganTimestamp').Timestamp_organ
    # Raw Intestine timestamp as index, and only alarms
    pc = pc.reset_index().set_index('mc_timestamp_organ')
    pc = pc[pc_alm]

    # Raw Intestine timestamp as index, and only alarms.
    # Create empty columns for alarms seen in pc that
    # are not in mc.
    mc = mc.set_index('Timestamp_organ')
    mc = mc[mc_alm]
    for alm in pc_alm:
        if not alm in mc:
            mc[alm] = 0

    pc, mc = pc.align(mc, fill_value=0)
    pc, mc = pc.astype('bool'), mc.astype('bool')
    return pc, mc


def show_overall_mean_difference(pc, mc):
    pc, mc = alarms_only(pc, mc)

    tpc, tmc = pc.mean().align(mc.mean(), join='outer')
    means = pd.DataFrame({'pc':tpc, 'mc':tmc}, index=tpc.index)
    means['diff'] = means['pc'] - means['mc']
    means = means.dropna()

    plotcat.categorical_value_differences(means.mc, means.pc)

def _read_or_infer_datetime(string, relevant_datetime_series):
    try:
        return pd.to_datetime(string, format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        res = pd.to_datetime(string, format='%H:%M:%S')
        mindate = relevant_datetime_series.min().date()
        maxdate = relevant_datetime_series.max().date()
        if not mindate == maxdate:
            raise ValueError('Cannot infer date - too many present in data.'
                             'min: {}, max: {}'.format(mindate, maxdate))
        return pd.datetime.combine(mindate, res.time())

def show_overlap_aligned(pc, mc, start=0, length_or_end=100,
                         **dvo_kwargs):
    """pc and mc are aligned dataframes
    start is either an integer (row number) or a string (datetime)
    length_or_end is either an integer (row numbers, or seconds) or a
        string (like '10 min', '5 s', or '12:45:43')

    dvo_kwargs are keyword arguments
        for utils.plotting.categorical.dummy_variable_overlaps
    """
    if isinstance(start, str):
        start = _read_or_infer_datetime(start, pc.index)

        if isinstance(length_or_end, str):
            try:
                end = _read_or_infer_datetime(length_or_end, pc.index)
            except ValueError:
                end = start + pd.to_timedelta(length_or_end)
        else:
            end = start + pd.to_timedelta(length_or_end, unit='s')

        pc_sub = pc[start:end]
        mc_sub = mc[start:end]
    else:
        end = start + length_or_end
        pc_sub = pc.iloc[start:end]
        mc_sub = mc.iloc[start:end]

    kwargs = {'drop_empty_rows':False,
              'drop_empty_cols':False,}
    kwargs.update(dvo_kwargs)
    plotcat.dummy_variable_overlaps(pc_sub, mc_sub, 'pc', 'mc',
                                    x_label='AlmNr', y_label='Time',
                                    **kwargs)

def show_overlap(pc, mc, start=0, length_or_end=100,
                 **dvo_kwargs):
    """pc and mc are unaligned dataframes
    start is either an integer (row number) or a string (datetime)
    length_or_end is either an integer (row numbers, or seconds) or a
        string (like '10 min', '5 s', or '12:45:43')

    dvo_kwargs are keyword arguments
        for utils.plotting.categorical.dummy_variable_overlaps
    """
    pc, mc = get_aligned_alarms(pc, mc)
    show_overlap_aligned(pc, mc, start, length_or_end,
                         **dvo_kwargs)

def rows_to_right_spot(alarm, df):
    """alarm is a single row (as a series) from AlmHist, with
    DatoTid as index.
    df is consolidated data.

    Returns the number of rows that df is off in attributing the alarm.
    """
    ts = 'Timestamp_organ' if 'Timestamp_organ' in df else 'OrganTimestamp'
    alarm_time = alarm.name
    alarm_col = 'E{}'.format(alarm.AlmNr)
    if not alarm_col in df:
        return None

    df_later =   (df[df[ts] >= alarm_time][alarm_col] == 1)
    df_earlier = (df[df[ts] <  alarm_time][alarm_col] == 1)

    df_later.index = range(len(df_later))
    df_earlier.index = reversed(range(len(df_earlier)))

    df_later = df_later[df_later]
    df_earlier = df_earlier[df_earlier]

    later = df_later.index[0] if len(df_later) > 0 else None
    earlier = -df_earlier.index[0] if len(df_earlier) > 0 else None

    if later is not None:
        if earlier is not None:
            if abs(earlier) < later:
                return earlier
            return later
        return later
    return earlier

def rows_to_right_spot_overall(alarms, df):
    """Returns a Series contiaining the number of rows to the right spot
    for every alarm in alarms.
    """
    off = pd.Series(map(lambda x: rows_to_right_spot(x[1], df), list(alarms.iterrows())))
    return off