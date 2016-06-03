import os

import numpy as np
import pandas as pd

import tbtools.dev as tbdev
import tbtools.iter as tbiter
import utils.data as ud
import utils.features as uf


def construct(sample_step, lag, dcwindow, rn, boawindow):
    """
    Constructs a design matrix with split indicator column, and stores
    it on disk.

    sample_step is the interval between indices in the design matrices
        Must be usable in pd.to_timedelta
    lag is the amount of time we are trying to predict over
        Must be usable in pd.to_timedelta
    dcwindow is the window size for delta C
        Must be usable in pd.to_timedelta
    rn is the number of carcasses taken into account for R, an int
    boawindow is the window size for bag-of-alarms
        Must be usable in pd.to_timedelta
    """
    path = ud.design_matrices.get_path(
        sample_step=sample_step,
        lag=lag,
        dcwindow=dcwindow,
        rn=rn,
        boawindow=boawindow)

    if os.path.isfile(path):
        print('Dump with those settings already exist')
        if input('Proceed anyway? y/n > ') != 'y':
            print('Aborting...')
            return
        else:
            print('Proceeding, overwriting old files')

    # Create sample times
    # We exploit that each day is a contiguous set of samples
    # and that `splits` has a row for each day
    splits = ud.design_matrices.get_splits()

    samples = None
    for i in splits.index:
        res = pd.date_range(splits.loc[i, 'min'],
                            splits.loc[i, 'max'],
                            freq=sample_step)
        if samples is None:
            samples = res
        else:
            samples = samples.union(res)
    samples = samples.sort_values()

    # Sample the features
    df = uf.sample_all(samples,
                       lag=lag,
                       delta_c_kwargs={'window': dcwindow},
                       r_kwargs={'n': rn},
                       bag_of_alarms_kwargs={'window': boawindow})
    # Compute and store total alarm count
    almcols = [c for c in df.columns if c.startswith('almnr_')]
    df['Alarm count'] = df[almcols].sum(axis=1)

    # Ensure order so we can select rows by date
    df = df.sort_index()

    # Assign split labels
    df['split'] = 'train'
    for k in splits['split'].unique():
        if k == 'train':
            continue
        dates = splits[splits['split'] == k]['date'].dt.date
        sd = np.in1d(df.index.date, dates)
        idx = df.index[sd]
        df.loc[idx, 'split'] = k

    df.to_pickle(path)
    print('Stored in', path)
