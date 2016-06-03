import pandas as pd

from ... import data
from . import alarms
from . import hangers
import tbtools.strings as tbs

def evaluate(df, name='consolidated table'):
    timecol = df['Timestamp_tx14'] if 'Timestamp_tx14' in df else df['Timestamp']
    date_start = str(timecol.iloc[0].date())
    date_end = str(timecol.iloc[-1].date())
    data.raw.set_verbosity(3)

    print('\nSummarization of {}'.format(name))
    with tbs.indent():
        print('Shape:',df.shape)
        print('HangerIDs:')
        with tbs.indent():
            bbh = data.raw.get('BitBusHist', date_start, date_end)
            bbh = bbh[bbh.Tx == 14]
            bbh = bbh[bbh.HangerID != 4098]

            a, b = bbh.HangerID.value_counts().align(df.HangerID.value_counts(), fill_value=0)
            misses = sum(a-b)
            fraction = misses / len(bbh)
            print('{:.3f}% of HangerIDs missed'.format(fraction))

        print('Alarms:')
        with tbs.indent():
            alm = data.raw.get('AlmHist', date_start, date_end)
            alm = alm[alm.AlmNr.isin(range(6006, 6350)) & (alm.AlmState==1)]

            rtrso = alarms.rows_to_right_spot_overall(alm, df)
            nans = sum(rtrso.isnull())
            vc = rtrso.value_counts()
            n_correct = vc[0]
            off = vc.drop(0)
            txt = str(off.iloc[:5])
            txt = txt[:txt.find('\ndtype:')]
            print('{:,} alarms hit spot on.'.format(n_correct))
            print('{:,} alarms completeley missed.'.format(nans))
            print('Partial misses (distance in rows, count of time it happened)')
            print(txt)
            if len(off) > 5:
                print('and {:,} more that are off.'.format(sum(off.iloc[5:])))

        print('Veterinarian data:')
        with tbs.indent():
            vcols = [x for x in df.columns if x.startswith('V') and x[1:].isdigit()]
            if len(vcols) == 0:
                print('None')
            else:
                print('Got {} columns of veterinarian data'.format(len(vcols)))
                with tbs.indent():
                    has_data = [c for c in vcols if df[c].sum() > 0]
                    print('{} of those have one or more 1s'.format(len(has_data)))
                    if 'V930' in has_data: print('V930 is part of them.')

def stats(df_list, df_names):
    return pd.DataFrame(list(map(evaluate_table, df_list)), index=df_names)

def evaluate_table(df):
    timecol = df['Timestamp_tx14'] if 'Timestamp_tx14' in df else df['Timestamp']
    date = str(timecol.iloc[0].date())
    data.raw.set_verbosity(3)

    stats = {}

    stats['Shape'] = df.shape

    bbh = data.raw.get('BitBusHist', date)
    bbh = bbh[bbh.Tx == 14]
    bbh = bbh[bbh.HangerID != 4098]

    a, b = bbh.HangerID.value_counts().align(df.HangerID.value_counts(), fill_value=0)
    misses = sum(a-b)
    fraction = misses / len(bbh)
    stats['HangerIDs missed percent'] = fraction*100

    alm = data.raw.get('AlmHist', date)
    alm = alm[alm.AlmNr.isin(range(6006, 6350)) & (alm.AlmState==1)]

    rtrso = alarms.rows_to_right_spot_overall(alm, df)
    nans = sum(rtrso.isnull())
    vc = rtrso.value_counts()
    n_correct = vc[0]
    off = vc.drop(0)

    stats['alarms_correct'] = n_correct
    stats['alarms_missed'] = nans
    stats['alarms_partial_misses'] = off.sum()

    vcols = [x for x in df.columns if x.startswith('V') and x[1:].isdigit()]
    stats['vet_data'] = len(vcols)
    has_data = [c for c in vcols if df[c].sum() > 0]
    stats['vet_data_active'] = len(has_data)
    stats['vet_v930_active'] = ('V930' in has_data)

    return stats