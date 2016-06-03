import pandas as pd
import numpy as np

from util import DataFrameBase

db_parent = os.environ['HOME'] + '/Speciale/data/exported/'
dbs = ('NN_1_10', 'NN_26', 'NN_CLC')

# https://docs.python.org/3/library/codecs.html#standard-encodings
global_settings = {
    'delimiter':';', 
    'encoding':'cp1252',
    'infer_datetime_format':True,
}

def parse_dates(*args):
    return {'parse_dates': list(args)}

datotid = parse_dates('DatoTid')
timestamp = parse_dates('Timestamp')

table_settings = {
    'AlmHist': datotid,
    'AlmTime': datotid,
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

dbs = {'clc': DataFrameBase(db_parent, 'clc', 
                global_settings=global_settings, 
                table_settings=table_settings),
        '1': DataFrameBase(db_parent, '1', 
                   global_settings=global_settings, 
                   table_settings=table_settings),
        '2': DataFrameBase(db_parent, '2', 
                   global_settings=global_settings, 
                   table_settings=table_settings)
        }

def normalize_df(df, intended_index=None):
    """df is a dataframe
    intended_index is a column name; this column will
        be used as index in the normalized df. If not
        defined, no new index is set.
    
    Discards rows that have all the same information, 
        including timestamps.
    Discards rows and column with only NaNs.
    """
    
    df = df.dropna(0, 'all').dropna(1, 'all')
    df.drop_duplicates(inplace=True)
    if intended_index is not None:
        df.set_index(intended_index, verify_integrity=True, inplace=True)
    df.sort_index(inplace=True)
    
    return df

def convert_db(db, target):
    # Normalize all tables, store them in nicer format.
    for tab in db.tables:
        intended_index = table_settings.get(tab, None)
        df = normalize_df(db[tab], intended_index)
        


def convert_all(target):
    for name in dbs:
        path = os.path.join(target, name)
        os.mkdirs(path)
        convert_db(dbs[name], path)