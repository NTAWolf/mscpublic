"""This module is a simple utility to preprocess and load the data files
in 'Dataset 15-21 MAJ 2012/'. 

Dataframe names are
    almhist
    organdata
    bitbushist
    raavare
    efterkontrolhist
"""

from collections import namedtuple
import pandas as pd

def load(path):
    """path is the (relative) path to the directory containing
    the .rpt files.

    Returns a namedtuple with almhist, organdata, bitbushist, 
        raavare, and efterkontrolhist
    """

    # path = 'data/Dataset 15-21 MAJ 2012/'
    files = [
        'AlmHist 15-21 MAJ 2012.rpt',
        'OrganData 15-21 MAJ 2012.rpt',
        'BitBusHist 15-21 MAJ 2012.rpt',
        'RAAVARE 15-21 MAJ 2012.rpt',
        'EfterkontrolHist 15-21 MAJ 2012.rpt',
    ]

    ## Almhist
    # - Some lines contain a comma in their text. The extraneous comma is
    # always followed by a space, so a simple search-replace ', ' to ' ' handles
    # it.
    # - The last three lines are unnecessary

    cleaned = 'almhist.csv'

    with open(path + files[0], 'r') as f:
        data = f.readlines()
        
    # Drop three last lines
    data = data[:-3]

    # Get rid of commas in text
    for i, line in enumerate(data):
        data[i] = line.replace(', ', '; ')
        
    with open(path + cleaned, 'w') as f:
        f.writelines(data)

    almhist = pd.read_csv(path + cleaned,
                          index_col=3,
                          parse_dates=[3],
                          infer_datetime_format=True,)

    ## Organdata
    organdata = pd.read_csv(path + files[1],
                          engine='python',
                          skipfooter=3,
                          index_col=5,
                          parse_dates=[5],
                          infer_datetime_format=True,)

    ## BitBusHist
    bitbushist = pd.read_csv(path + files[2],
                             engine='python',
                             skipfooter=3,
                             index_col=3,
                             parse_dates=[3],
                             infer_datetime_format=True,)

    ## Raavare
    raavare = pd.read_csv(path + files[3],
                          engine='python',
                          skipfooter=3,
                          index_col=3,
                          parse_dates=[3, 75, 106],
                          infer_datetime_format=True,)

    ## EfterkontrolHist
    efterkontrolhist = pd.read_csv(path + files[4],
                                   engine='python',
                                   skipfooter=3,
                                   index_col=3,
                                   parse_dates=[3],
                                   infer_datetime_format=True,)

    return namedtuple(
            'RPTFiles', ['almhist', 'organdata', 
                         'bitbushist', 'raavare', 
                         'efterkontrolhist'])(
        almhist=almhist,
        organdata=organdata,
        bitbushist=bitbushist,
        raavare=raavare,
        efterkontrolhist=efterkontrolhist,
    )

