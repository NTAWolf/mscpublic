from functools import lru_cache
import os
import pickle

import pandas as pd

from . import paths
import tbtools.strings as tbstr


def get_splits():
    """
    Returns the dataframe containing split and sample
    times information.
    """
    splitpath = paths.get_file('design_matrices', 'splits.pickle')
    return pd.read_pickle(splitpath)


def get_path(sample_step, lag, dcwindow, rn, boawindow):
    """
    Based on the passed settings, returns the full path to
    what may or may not be an existing design matrix pickle.
    """
    name = '  '.join(['ss {}', 'L {}', 'dcw {}',
                      'rn {}', 'bw {}']
                     ).format(sample_step, lag, dcwindow,
                              rn, boawindow)
    path = os.path.join(paths.Paths.design_matrices, name)
    return path


def get_by_settings(sample_step, line, lag, dcwindow, rn, boawindow,
                    add_intercept=False, return_dicts=True):
    """
    Returns the design matrix matching the given settings.
    If return_dicts is True, returns a dict that can be indexed as
        res['train']['x']
    """

    path = get_path(sample_step, lag, dcwindow, rn, boawindow)

    if os.path.isfile(path):
        print('Loading from', path)
    else:
        print('Constructing...')
        import utils.design_matrices as udm
        udm.construct(sample_step, lag, dcwindow, rn, boawindow)

    res = pd.read_pickle(path)

    if line is not None:
        res = res[res.line == line].drop('line', axis=1)
        alm = get('almnr_lines')
        almnr = alm.AlmNr[alm['L{}_soft'.format(line)]].values
        keepcol = [c for c in res if (
            (not c.startswith('almnr_')) or
            (int(c[6:]) in almnr)
        )]
        res = res[keepcol]
    if add_intercept:
        res['intercept'] = 1

    if return_dicts:
        splits = {k: res[res['split'] == k].drop('split', axis=1)
                  for k in res['split'].unique()}
        splits = {k: {'x': (v.drop('y', axis=1)),
                      'y': v['y']
                      } for k, v in splits.items()}

        return splits

    return res


def get(tablename):
    """Uses fuzzy matching to locate the requested table and db.
    Returns a pandas DataFrame.
    """
    path = paths.get_file('design_matrices', tablename)
    return pd.read_pickle(path)
