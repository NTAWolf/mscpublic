import os
import utils.data.unused as udu

db_parent = os.environ['HOME'] + '/Speciale/data/exported/'

def parse_dates(*args):
    return {'parse_dates': list(args)}

datotid = parse_dates('DatoTid')
timestamp = parse_dates('Timestamp')

global_settings = {
    'delimiter':';', 
    'encoding':'cp1252',
    'infer_datetime_format':True,
}

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

def get_db(name):
    return udu.DataFrameBase(db_parent, name, 
                    global_settings=global_settings, 
                    table_settings=table_settings)
