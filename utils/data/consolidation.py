from functools import lru_cache
import pandas as pd
import numpy as np

import tbtools.iter as tbiter
import tbtools.panda as tbpd
import tbtools.strings as tbstr

from . import raw, enhanced
from ..evaluation.consolidation import evaluate

_SETTINGS = {
    'report': {
        'max_minutes_tx14_to_raavaredb': 120,
        'drop_missing_organ_timestamp': True, # I think?
        'tx14_od_min_diff_seconds': 8,
        'tx14_od_max_diff_seconds': 12.0,
        'alm_nr_subset': range(6006, 6320+1),
    },
    'preconsolidated': {
        'max_minutes_tx14_to_raavaredb': 120, # Double check, please
        'drop_missing_organ_timestamp': False,
        'tx14_od_min_diff_seconds': 7.4,
        'tx14_od_max_diff_seconds': 13.0,
        'alm_nr_subset': range(6006, 6347+1),
    },
    'enhanced': {
        'use_enhanced_bbh':True,
        'merge_reinspection_indicator':True,
    }
}

_SETTINGS['enhanced'].update(_SETTINGS['preconsolidated'])

class Logger:
    verbosity = 3

    def __init__(self, verbosity=3):
        self.verbosity = verbosity
        self.do_hold = False
        self.mem = []

    def log(self, verbosity, *args, **kwargs):
        if self.do_hold:
            self.mem.append((verbosity, args, kwargs))
            return
        if verbosity <= Logger.verbosity:
            print(*args, **kwargs)

logger = Logger()
log = logger.log

def set_verbosity(verbosity):
    logger.verbosity = verbosity

@lru_cache(4)
def get(time_selection=None, time_selection_end=None,
        db=None, settings='preconsolidated',
        integrity_check=False, do_evaluate=False,
        progress_bar=False):
    """Performs consolidation of the wanted DB in the wanted
    time range, and returns the resulting table.
    """

    log(1, 'Consolidating from db "{}" with settings: "{}"'.format(db, settings))

    if time_selection_end is None:
        log(2, 'for {}'.format(time_selection))
    else:
        log(2, 'for {} to {}'.format(time_selection, time_selection_end))

    table_kwargs = {'time_selection':time_selection,
                    'time_selection_end':time_selection_end,
                    'db':db}

    with tbstr.indent():
        raw.set_verbosity(logger.verbosity)
        consolidated = _Consolidator(table_kwargs, _SETTINGS[settings])\
                        .run(use_progress_bar=progress_bar)
        if integrity_check:
            test_integrity(consolidated)
        if do_evaluate:
            evaluate(consolidated)
    return consolidated


class _Consolidator:
    def __init__(self, data_settings, algorithm_settings):
        self.data_settings = data_settings
        self.settings = algorithm_settings
        self.indentation = 1

    def run(self, use_progress_bar=False):
        calls = (
            self.merge_bbh_rvd,
            self.drop_missing_intestine_timestamps,
            self.enforce_timelimit_tx14_rvd,
            self.drop_older_reinspections,
            self.merge_reinspection_indicator,
            self.merge_bemaerk,
            self.merge_od,
            self.drop_merge_errors,
            self.merge_alm,
            self.enforce_dtypes,
        )
        if use_progress_bar:
            captor = tbstr.Captor()
            pbar = tbiter.ProgressBar(len(calls))
            pbar.draw()
            i = 0
            for func in calls:
                i += 1
                if i >= 7:
                    print('DEBUG stop short before func {}'.format(func))
                    return self
                with captor:
                    func()
                pbar.increment()
            captor.flush()
        else:
            for func in calls:
                func()

        return self.consolidated

    def merge_bbh_rvd(self):
        log(3, 'Merging BitBusHist and RaavareDB')
        with tbstr.indent():
            if self.settings.get('use_enhanced_bbh', False):
                bbh = enhanced.get('bbh', self.data_settings['time_selection'],
                                   self.data_settings['time_selection_end'])
            else:
                bbh = raw.get('BitBusHist', **self.data_settings)
            rvd = raw.get('RaavareDB', keep=('Id',), **self.data_settings)
        tbpd.rename_col(rvd, Id='RaavareId')

        # Select where Tx==14, drop useless col Tx
        bbh = bbh[bbh.Tx == 14].drop(['Tx'], axis=1)
        # Select only bbh entries where HangerID is in RVD
        bbh = bbh[bbh.HangerID.isin(rvd.HangerID)]
        # Move timestamp out
        bbh = bbh.reset_index()


        cons =  rvd.reset_index().merge(bbh,
                                        on='HangerID',
                                        suffixes=('_rvd', '_tx14'),
                                        how='outer')
        # Move timestamp in for eased visual comparison
        cons = tbpd.move_col(cons, 'Timestamp_tx14', 0)

        self.consolidated = cons.sort_values(by=['Timestamp_tx14', 'Timestamp_rvd'])

    def drop_missing_intestine_timestamps(self):
        # Get rid of rows with missing timestamps
        if not self.settings['drop_missing_organ_timestamp']:
            return

        before = len(self.consolidated)
        self.consolidated = self.consolidated.dropna(
            subset=['Timestamp_tx14', 'Timestamp_rvd'], how='any')
        log(3, tbstr.red('Drop {} rows: Missing intestine remover '
               'timestamps').format(before - len(self.consolidated)))


    def enforce_timelimit_tx14_rvd(self):
        """Applies the time limit from tx14 to raavareDB (default 120 minutes).
        Violating rows are dropped.
        """
        df = self.consolidated
        diff = (df['Timestamp_rvd'] - df['Timestamp_tx14']).astype('timedelta64[m]')
        mask = (0 <= diff) & (diff <= self.settings['max_minutes_tx14_to_raavaredb'])
        labels = df.index[~mask]
        before = len(df)
        df.drop(labels, axis=0, inplace=True)
        log(3, tbstr.red('Drop {} rows: timestamps tx14 and raavaredb differ by '
               'more than {} minutes.').format(
                before - len(df),
                self.settings['max_minutes_tx14_to_raavaredb']))
        self.consolidated = df

    def drop_merge_errors(self):
        """Drop rows that are merged so that the Tx14 timestamp precedes
        the intestine remover timestamp. Also, remove rows that exceed
        2 hours from intestine remover to Tx14.
        """
        df = self.consolidated
        len_before = len(df)
        diff = (df.Timestamp_tx14 - df.Timestamp_organ).astype('timedelta64[s]')
        # organ must be before tx14
        # 2 hours (2 hours of 60 minutes of 60 seconds) seem a reasonable limit
        self.consolidated = df[(0 <= diff) & (diff <= 2*60*60)]
        log(3, tbstr.red('Drop {} rows: Duplicates due to merge with '
                   'intestine remover data.'.format(len_before - len(self.consolidated))))

    def drop_older_reinspections(self):
        """Drop duplicate hangerids; keep only the last encounter with one
        As the hangers go around, they may enter RaavareDB multiple times. This
        makes sure we only take the last encounter into account.
        """
        # This method relies on the outer merge made in previous steps,
        # so that rows with same tx14 timestamp and hangerid definitely
        # are duplicates, where we should keep the last one.
        df = self.consolidated.sort_values(by=['Timestamp_tx14', 'Timestamp_rvd'])
        len_before = len(df)
        df.drop_duplicates(subset=['Timestamp_tx14', 'HangerID'], keep='last', inplace=True)
        log(3, tbstr.red('Drop {} rows: Duplicates in merged data due to merging or reinspections.')\
            .format(len_before - len(df)))
        self.consolidated = df

    def merge_reinspection_indicator(self):
        if not self.settings.get('merge_reinspection_indicator', False):
            return

        log(3, 'Merging with reinspection indicator')
        with tbstr.indent():
            ri = enhanced.get('reinspection')
        df = self.consolidated
        len_before = len(df)
        df = df.merge(ri, how='inner').sort_index()
        len_diff = len_before - len(df)
        if len_diff != 0:
            print('Uh oh, lost {} rows in merge with '
                  'reinspection indicator'.format(len_diff))
        self.consolidated = df

    def merge_bemaerk(self):
        log(3, 'Merging with BemaerkningKode')
        with tbstr.indent():
            bmk = raw.get('BemaerkningKode', **self.data_settings)
        bmk = bmk.drop(['Id'], axis=1)
        bmk = bmk.reset_index()
        bmk = pd.get_dummies(bmk, prefix='V', prefix_sep='', columns=['Kode'])
        bmk = bmk.groupby(['Timestamp', 'RaavareId']).sum()
        assert not ((bmk > 1) | (bmk < 0)).any().any(),\
               "BemaerkningKode sums per RaavareId lt or gt 1"
        bmk = bmk.reset_index()
        tbpd.rename_col(bmk, Timestamp='Timestamp_bmk')

        df = self.consolidated
        df = df.merge(bmk, left_on='RaavareId', right_on='RaavareId', how='outer')
        vcols = [c for c in df if c[0]=='V' and c[1:].isdigit()]
        df.loc[:,vcols] = df.loc[:,vcols].fillna(0)

        self.consolidated = df

    def merge_od(self):
        log(3, 'Merging with OrganData')
        with tbstr.indent():
            log(4, 'in time range {}-{}'.format(
                self.settings['tx14_od_min_diff_seconds'],
                self.settings['tx14_od_max_diff_seconds']))
            od = raw.get('OrganData', **self.data_settings)
        df = self.consolidated.sort_values(by=['Timestamp_tx14', 'Timestamp_rvd'])
        candidates = tbpd.match_in_diff_range(ahead=df.Timestamp_tx14,
                                              behind=od.reset_index()['Timestamp'],
                                              min_diff=pd.Timedelta(
                                                seconds=self.settings['tx14_od_min_diff_seconds']),
                                              max_diff=pd.Timedelta(
                                                seconds=self.settings['tx14_od_max_diff_seconds']),
                                              quiet=(logger.verbosity <= 3))

        candidates = list(map(lambda x: df.HangerID[x], candidates))

        # print stats
        ser = pd.Series([len(x) for x in candidates])
        with tbstr.indent():
            log(5, 'Number of matches by OrganData in consolidated HangerID:')
            ser = str(ser.value_counts())
            with tbstr.indent():
                log(5, ser)
        # end print stats

        od = od.reset_index()
        od['HangerID'] = [np.nan if len(x)==0 else x.values[0] for x in candidates]
        od.dropna(axis=0, subset=['HangerID'], inplace=True)
        tbpd.rename_col(od, Timestamp='Timestamp_organ')

        df = pd.merge(df, od, on='HangerID', suffixes=('','_organ'))
        df = tbpd.move_col(df, 'Timestamp_organ', 2)

        self.consolidated = df.sort_values(by=['Timestamp_tx14', 'Timestamp_rvd'])

    def merge_alm(self):
        log(3, 'Merging df with AlmHist')
        with tbstr.indent():
            alm = raw.get('AlmHist', **self.data_settings)
        # Use only alarms when they are raised (AlmState)
        alm = alm[(alm.AlmState == 1) & \
                  (alm.AlmNr.isin(self.settings['alm_nr_subset']))]
        alm = alm[['AlmNr']].sort_index().AlmNr
        almnr_seen = alm.unique()
        almnr_to_col = lambda x: 'E{}'.format(x)

        dummy = pd.DataFrame(np.zeros((len(self.consolidated), len(almnr_seen))),
                             columns=list(map(almnr_to_col, almnr_seen)))
        ts = self.consolidated.Timestamp_organ

        # This is really slow, and runs on only one kernel. Divide and conquer?
        # def worker(ran)

        def insert_alarms(previous, current, index):
            raised = alm[previous:current]
            for v in raised:
                colname = almnr_to_col(v)
                dummy[colname].iloc[index] = 1

        # Special handling for the first one:
        # Assume that alarms from 9 seconds before (judged from
        # differences in organtimestamps) belong to the first
        # carcass.
        current = ts.iloc[0]
        previous = current - pd.to_timedelta('9 sec')
        insert_alarms(previous, current, 0)

        for index in range(1, len(dummy)):
            previous, current = ts.iloc[index-1], ts.iloc[index]
            insert_alarms(previous, current, index)

        dummy.index = self.consolidated.index

        self.consolidated = pd.concat([self.consolidated, dummy], axis=1)

    def enforce_dtypes(self):
        if self.settings.get('use_enhanced_bbh', False):
            self.consolidated['uid'] = self.consolidated['uid'].astype(int)
        if self.settings.get('merge_reinspection_indicator', False):
            self.consolidated['reinspected'] = self.consolidated['reinspected'].astype(int)


def test_integrity(df):
    print('\nIntegrity report:')

    problem = False

    with tbstr.indent():
        if not all(  (df.Timestamp_organ < df.Timestamp_tx14) \
                   & (df.Timestamp_tx14 < df.Timestamp_rvd)):
            problem = True
            print("Timestamps for a single row must be in the order "
                  "organ - tx14 - raavaredb. One or more entries violate this.")

        diff_too_small = []

        dupes = df[df.duplicated(subset=['HangerID'], keep=False)]
        for hangerid, v in dupes.groupby('HangerID'):
            tx14 = v.Timestamp_tx14

            mindiff = min(map(lambda x: max(x)-min(x), tbiter.pairwise(tx14)))
            if mindiff.total_seconds() < 2*60*60:
                problem = True
                diff_too_small.append(hangerid)

        if 0 < len(diff_too_small):
            print(("Tx14 time difference between duplicates "
                   "too small for {}").format(diff_too_small[:50]))
            if len(diff_too_small) > 50:
                print('... and {} more.'.format(len(diff_too_small) - 50))

        if not problem:
            print("No problems found.")

def _debug_track_hangerid(date, hid):
    table_kwargs = {'time_selection':date,
                    'time_selection_end':None,
                    'db':None}

    cons = _Consolidator(table_kwargs, _SETTINGS['preconsolidated'])

    steps = (
        cons.merge_bbh_rvd,
        cons.drop_missing_intestine_timestamps,
        cons.enforce_timelimit_tx14_rvd,
        cons.drop_older_reinspections,
        cons.merge_od,
        cons.drop_merge_errors,
        cons.merge_alm,
    )

    for step in steps:
        step()
        if not hid in cons.consolidated.HangerID:
            print('HangerID is dropped in ', step.__name__, step)
            return

    print("No error found!")
