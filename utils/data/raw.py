from functools import lru_cache
import os
import re

import pandas as pd
import numpy as np

from tbtools.dev.html import display_table
from tbtools.strings import fuzzy_match, indent, red

from .paths import Paths

USE_EXPORTED_DATABASE = Paths.raw_dbs[0]
GHOST_HANGERS = (4098,)


GLOBAL_SETTINGS = {
    'delimiter':';',
    'encoding':'cp1252',
    'infer_datetime_format':True,
}

_VERBOSITY = 3
def log(verbosity, *args, **kwargs):
    if verbosity <= _VERBOSITY:
        print(*args, **kwargs)

def set_verbosity(verbosity):
    global _VERBOSITY
    _VERBOSITY = verbosity

def parse_dates(*args):
    return {'parse_dates': list(args)}

datotid = parse_dates('DatoTid')
timestamp = parse_dates('Timestamp')

TABLE_SETTINGS = {
    'AlmHist': datotid,
    'AlmTime': datotid,
    'BemaerkningKode': timestamp,
    'BitBusHist': timestamp,
    'BitHistCheck': timestamp,
    'Efterkontrol': timestamp,
    'EfterkontrolHist': timestamp,
    'HangerFlow': timestamp,
    'LEVERANDOR$': parse_dates('DATO_TID'),
    'LineSpeed': timestamp,
    'OrganData': timestamp,
    'OrganDataHist': timestamp,
    'ProdDataTemp': parse_dates('Starttime', 'Endtime'),
    'RaavareDB': timestamp,
}

def get(tablename, time_selection=None, time_selection_end=None, db=None, keep=tuple()):
    """Uses fuzzy matching to locate the requested table and db.
    Returns a pandas DataFrame.
    """
    tablename = fuzzy_match(tablename, TABLE_SETTINGS.keys())
    dbpath = fuzzy_match(db, Paths.raw_dbs) if db else USE_EXPORTED_DATABASE
    return get_specific(dbpath, tablename, time_selection, time_selection_end, keep=keep)

@lru_cache(maxsize=8)
def get_specific(dbpath, tablename, time_selection=None, time_selection_end=None, keep=tuple()):
    csv_kwargs = GLOBAL_SETTINGS.copy()
    tabsettings = TABLE_SETTINGS.get(tablename, {})
    csv_kwargs.update(tabsettings)

    tablename += '.csv'
    path = os.path.join(dbpath, tablename)
    df = pd.read_csv(path, **csv_kwargs)
    log(1, 'Getting {}:{}'.format(_str_path_name(dbpath), _str_path_name(path)))

    if 'parse_dates' in tabsettings:
        indexcol = tabsettings['parse_dates'][0]
        df = df.set_index(indexcol).sort_index()


    drop_nans_and_monovalues(df)
    drop_low_median_lag_var(df, keep=keep)
    drop_ghost_hangers(df)

    if time_selection is not None:
        if time_selection_end is not None:
            df = df[time_selection:time_selection_end]
        else:
            df = df[time_selection]

    with indent():
        log(2, ('Earlist index: {}\n'
                '   Last index: {}').format(df.index.min(), df.index.max()))

    return df

def drop_nans_and_monovalues(df):
    """Modifies DataFrame df inplace, printing to stdout
    what modifications are made.
    """
    nan_cols = get_nans(df)
    for c in nan_cols:
        with indent():
            log(3, red('Dropping {}, as it is all null'.format(c)))

    monoval_cols = get_monovalues(df)
    for c in monoval_cols:
        with indent():
            log(3, red('Dropping {}; has only one value: {}'.format(c, df[c].unique()[0])))

    df.drop(nan_cols + monoval_cols, axis=1, inplace=True)

def get_nans(df):
    """Returns a list of columns in df which
    contain only nans
    """
    nan_cols = []
    for col in df:
        if df[col].isnull().all():
            nan_cols.append(col)
    return nan_cols

def get_monovalues(df):
    """Returns a list of columns in df which
    contain only a single value
    """
    monoval_cols = []
    for col in df:
        unique = df[col].unique()
        if len(unique) == 1 and not df[col].isnull().any():
            monoval_cols.append(col)

    return monoval_cols

def drop_ghost_hangers(df):
    """Removes rows where HangerID is in the
    list of GHOST_HANGERS. This is done in-place.

    The action is printed to stdout.
    """

    hid = 'HangerID'
    if hid in df:
        ghosts = df[hid].isin(GHOST_HANGERS)
        if not ghosts.any():
            return
        ghost_indices = df.index[ghosts]
        with indent():
            log(3, red('Dropping {:,} of {:,} rows ({:.2f} %); ghost '
                    'hanger (HangerID in {})').format(len(ghost_indices),
                                                      len(df),
                                                      100*len(ghost_indices)/len(df),
                                                      GHOST_HANGERS))
        df.drop(ghost_indices, axis=0, inplace=True)

def drop_low_median_lag_var(df, threshold=1e-3, keep=tuple()):
    to_drop = get_low_median_lag_var(df, threshold)
    to_drop = [c for c in to_drop if not c in keep]
    for col in to_drop:
        with indent():
            log(3, red('Dropping {}; low median lag '
                   'standard deviation.'.format(col)))
    df.drop(to_drop, axis=1, inplace=True)

def get_low_median_lag_var(df, threshold=1e-3):
    to_drop = []
    for col in df:
        unq = df[col].unique()
        if len(unq) == 2 or \
           (len(unq) == 3 and any(np.isnan(unq))):
            # Don't drop binary columns
            continue
        try:
            mls = median_lag_var(df[col])
        except TypeError as te:
            pass
        else:
            if mls < threshold:
                print('col {}, mls {}'.format(col, mls))
                to_drop.append(col)
    return to_drop

def median_lag_var(ser, window=3):
    """Calculates the variance of
    the rolling median (size window) of
    the difference between each consecutive value in ser.
    """
    rolling_median = pd.Series(ser.values[1:]-ser.values[:-1]).rolling(window).median()
    return rolling_median.dropna().var()


TABLE_DESCRIPTIONS = {
    'AlmHist': {
        'description': 'All alarms through the ages.',
        'table':[
            ['Key', 'Contents'],
            ['ServerName', 'Only one value, NBLA3150 (for Sept 28)'],
            ['AlmNr', 'Alarm type ID. Used for selecting a range of alarms pertaining to OrganData.'],
            ['AlmState', 'Two values: 1 and 2. 1 is for the alarm before being acknowledged, 2 is the alarm upon being acknowledged. Thus almost every alarm should be duplicated in the table.'],
            ['TagNavn', 'Overall alarm name. L1 L2 indicates which line the alarm is from.'],
            ['BitNr', '?? Following the power law'],
            ['TB1', 'Device name'],#Sub-title. About half of these are contained in TagNavn.'],
            ['TB2', 'Device code'],#?? Following power law. Are in the style U0101, C0313. Some sort of location indicator.'],
            ['TB3', 'Alarm details'],
            ['TB4', 'Empty, dropped.'],
            ['TB5', 'String for human-readable location'],
            ['Class', 'Refer to AlmClass table. Only five classes are present. 1 is alarm, 2 is warning, 3 is blocked alarm, 5 is OS Process control system messages, 11 is auto-acknowledged'],
            ['Typ', 'Numbers probably referring to AlmType. 19 is warning high, 1 is alarm high, 2 is alarm low.'],
            ['TimeDiff', 'Nonzero only for AlmState==2. Represents time between alarm raised and acknowledged.'],
        ]},
    'BitBusHist': {
        'description': 'HangerID checkpoints (Tx) with timestamps',
        'table':[
            ['Key', 'Contents'],
            ['Id', 'Unik nøgle'],
            ['Tx', 'Id-læser position'],
            ['HangerId', 'Hængejerns id'],
            ['Timestamp', 'Tidsstempel: YYYY-MM-DD hh:mm:ss'],
        ]},
    'RaavareDB': {
        'description': 'Veterinarian info about quality and cutoffs.',
        'table':[
            ['Key', 'Contents'],
            ['Hængejerns ID', 'Nøgle'],
            ['Timestamp', 'Tidsstempel: YYYY-MM-DD hh:mm:ss'],
            ['Vaegt_Maskinskade', 'Vægt på fraskæring pga. maskinskade (1/10 kg)'],
            ['Vaegt_Bemaerkning', 'Vægt på fraskæring pga. gødningsforurening (1/10 kg)'],
            ['Vaegt_Afregning', 'Afregningsvægt (1/10 kg)'],
            ['Vaegt_Ukorr', 'Ukorrigeret vægt (1/10 kg)'],
            ['LeverandorNr', 'Leverandør nummer'],
            ['Art', 'Art'],
            ['Køn', 'Køn'],
            ['Kødpct.', 'Kød procent (1/10 %)'],
            ['Kode', 'Bemærkningskode'],
            ['Id', 'ID to match with entries in BemaerkningKode'],
        ]},
    'OrganData': {
        'description': 'Data from the intestine remover',
        'table':[
            ['Key', 'Contents'],
            ['Id', 'Unik nøgle'],
            ['Linje', 'Id-læser position '],
            ['Orientering', 'Status på orientering '],
            ['Behandlet', 'Status på process'],
            ['Timestamp', 'Tidsstempel: YYYY-MM-DD hh:mm:ss'],
            ['Albuemaal', 'Mål i mm.'],
            ['Laengdemaal', 'Mål i mm.'],
            ['HangerID', 'Occasionally present. Values are dubious; discard.']
        ]},
    'BemaerkningKode': {
        'description': 'Data from the veterinarians',
        'table':[
            ['Key', 'Contents'],
            ['Id', 'Useless'],
            ['RaavareId', 'ID to match with RaavareDB'],
            ['Timestamp', 'Tidsstempel: YYYY-MM-DD hh:mm:ss'],
            ['Kode', 'Veterinarian code for e.g. dung contamination.']
        ]}
}

def describe(tablename):
    """Renders a table describing the keys of the
    given tablename.
    """
    val = TABLE_DESCRIPTIONS[tablename]
    desc = val['description']
    tab = val['table']

    print('{}: {}'.format(tablename, desc))
    display_table(tab)

def browse(database=None, table=None, order_by='name'):
    """Prints an overview of the given database
    (or the one in USE_EXPORTED_DATABASE if not defined),
    optionally restricted to a single table.
    Special value of database: 'all'.

    order_by is in {'name', 'length', None}
    """
    ORDER_SPECS = {'name':lambda x: sorted(x, key=lambda y: y[0]),
                   'length':lambda x: sorted(x, key=lambda y: y[1], reverse=True)}

    if database is None:
        return browse(USE_EXPORTED_DATABASE, table, order_by)

    if database == 'all':
        for p in Paths.raw_dbs:
            browse(p, table, order_by)
        return

    if not database in Paths.raw_dbs:
        return browse(fuzzy_match(database, Paths.raw_dbs), table, order_by)

    dbname = _str_path_name(database)
    tables = _table_paths(database)

    if table is not None:
        table = fuzzy_match(table, tables)
        tablename = _str_path_name(table)
        print('Database {}, 1 of {} tables:\n'.format(dbname, len(tables)))
        tables = [table]
    else:
        print('Database {}, {} tables:\n'.format(dbname, len(tables)))

    tables = [(_str_path_name(t), _table_len(t), _table_header(t)) for t in tables]
    if order_by is not None:
        func = ORDER_SPECS[order_by]
        tables = func(tables)

    for name, length, header in tables:
        print('{}\n{:,} rows'.format(name, length))
        with indent():
            print('{}'.format('\n\t'.join(header)))
        print()
    print()

def _str_path_name(path):
    path = path.strip().strip(os.path.sep)
    if path.endswith('.csv'):
        path = path[:-4]
    path = os.path.basename(path)
    return path

def _table_paths(db_dir_path):
    return [os.path.join(db_dir_path, x) for x in os.listdir(db_dir_path) if x.endswith('.csv')]

def _table_len(path, subtract_header=True):
    i = 0
    with open(path, 'rb') as f:
        while f.readline():
            i += 1
    return i-1 if subtract_header else i

def _table_header(path, separator=';'):
    with open(path, 'r', encoding=GLOBAL_SETTINGS['encoding']) as f:
        header = str(f.readline()).strip()

    if separator is not None:
        return header.split(separator)
    return header


