from functools import lru_cache
import os

import pandas as pd

import tbtools.filewrangling as tbfiles
import tbtools.strings as tbstr
from .paths import Paths

_PARSE_DATES = ['Timestamp', 'OrganTimestamp']
_VERBOSITY = 3


def log(verbosity, *args, **kwargs):
    if verbosity <= _VERBOSITY:
        print(*args, **kwargs)


def set_verbosity(verbosity):
    global _VERBOSITY
    _VERBOSITY = verbosity


@lru_cache(4)
def get(time_selection, time_selection_end=None, usecols=None, db='NN'):
    """Returns a dataframe containing all preconsolidated data
    for the given period.

    db is either 1 or 2. The first contains data for Sep, Oct, the seconds
        contains data for Nov, Dec.
    """

    log(1, 'Loading preconsolidated data')

    if time_selection_end is None:
        log(2, 'for {}'.format(time_selection))
    else:
        log(2, 'for {} to {}'.format(time_selection, time_selection_end))

    dates = _to_date_range(time_selection, time_selection_end)

    with tbstr.indent():
        df = _get_from_date_range(dates, usecols=usecols, db=db)

    if time_selection_end is not None and 'Timestamp' in df.columns:
        df = df.set_index('Timestamp').loc[
            time_selection:time_selection_end].reset_index()

    return df


def _to_date_range(time_selection, time_selection_end=None):
    if time_selection_end is not None:
        # Assume date uses separators if not only digits in the first chars
        L = 10 if not time_selection[:8].isdigit() else 8
        dates = pd.date_range(time_selection[:L], time_selection_end[:L])
    else:
        dates = pd.date_range(time_selection, time_selection)

    return dates


def _get_from_date_range(date_range, usecols=None, db=1):
    paths = _get_file_paths(date_range, db=db)
    filenames = ('B{:%Y-%m-%d}.csv'.format(day) for day in date_range)
    dfs = []
    for path in paths:
        skipfooter = tbfiles.find_beginning_of_end(
            path, lambda x: x == 'EOF', patience=2)
        if skipfooter > 0:
            with tbstr.indent():
                log(2, tbstr.red(
                    'Drop {} rows, as they are EOF'.format(skipfooter, path)))
        res = pd.read_csv(path, skipfooter=skipfooter, engine='python',
                          sep=';',
                          parse_dates=[d for d in _PARSE_DATES \
                                       if usecols is None or d in usecols],
                          usecols=usecols)
        dfs.append(res)

    return pd.concat(dfs, ignore_index=True)


def _get_file_paths(date_range, db):
    if db == 1:
        filenames = ('B{:%Y-%m-%d}.csv'.format(day) for day in date_range)
        bdir = Paths.preconsolidated_dir_1
    elif db == 2:
        filenames = (('NN.'
                      + ('{:%d. %B}.txt'.format(day).lower().strip('0')))
                     for day in date_range)
        bdir = Paths.preconsolidated_dir_2
    else:
        raise NotImplementedError()

    paths = []
    for f in filenames:
        path = os.path.join(bdir, f)
        if not os.path.exists(path):
            # log(2, 'No file found in {}'.format(path))
            continue
        else:
            paths.append(path)
            # log(1, 'Loading preconsolidated in {}'.format(path))
    return paths


def clean(df):
    """Corrects the (perceived) errors made when
    originally consolidating this data.
    """

    prev_len = len(df)
    null_rows = (df.isnull().sum(axis=1) > 0)
    log(2, 'Dataframe is {} rows'.format(len(df)))
    with tbstr.indent():
        log(3, '{} ghost hanger entries'.format(sum(df.HangerID == 4098)))
        log(3, '{} rows with NaNs'.format(sum(null_rows)))
        log(3, '{} rows overlap in the previous categories'.format(
            sum((df.HangerID == 4098) & (null_rows))
        ))
    df = df.dropna(axis=0)
    log(2, tbstr.red('Drop {} rows due to NaNs'.format(prev_len-len(df))))
    prev_len = len(df)
    df = df[df.HangerID != 4098]
    log(2, tbstr.red(
        'Drop {} rows due to ghost hangers'.format(prev_len-len(df))))

    dupes = df.duplicated(keep='last')
    if sum(dupes) > 0:
        log(2, tbstr.red(
            'Drop {} rows that are plain duplicates'.format(sum(dupes))))
        df.drop_duplicates(keep='last', inplace=True)

    return df
