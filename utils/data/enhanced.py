from functools import lru_cache
import os

import pandas as pd

from . import paths
from . import raw
import tbtools.strings as tbstr

TABLE_SETTINGS = {
    'uid_reinspection.csv':
        { 'dtype': {'uid':int, 'reinspected':int} },
    'bitbushist.csv':
        { 'dtype': {'uid':int} },
    'consolidated.csv':
        { 'parse_dates': ['Timestamp_tx14', 'Timestamp_rvd',
                          'Timestamp_organ', 'Timestamp_bmk']}
}

TABLE_SETTINGS['bitbushist.csv'].update(raw.TABLE_SETTINGS['BitBusHist'])

def get(tablename, time_selection=None, time_selection_end=None, silent=False):
    """Uses fuzzy matching to locate the requested table and db.
    Returns a pandas DataFrame.
    """
    path = paths.get_file('enhanced', tablename)
    return get_specific(path, time_selection, time_selection_end, silent)

@lru_cache(maxsize=8)
def get_specific(path, time_selection=None, time_selection_end=None, silent=False):
    if not silent:
        print('Getting {}'.format(path))

    settings = TABLE_SETTINGS.get(os.path.basename(path), {})
    df = pd.read_csv(path, **settings)

    if 'parse_dates' in settings:
        indexcol = settings['parse_dates'][0]
        df = df.set_index(indexcol).sort_index()

    if time_selection is not None:
        if time_selection_end is not None:
            df = df[time_selection:time_selection_end]
        else:
            df = df[time_selection]

    if not silent:
        with tbstr.indent():
            print(('Earlist index: {}\n'
                   '   Last index: {}').format(df.index.min(), df.index.max()))

    return df