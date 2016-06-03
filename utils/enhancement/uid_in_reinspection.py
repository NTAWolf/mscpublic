import os

import pandas as pd

from .. import data as ud
from ..data.paths import Paths
from ..plotting.maps import txgraph

import tbtools.strings as tbstr

REINSPECTION_TX = [tx for tx,d in txgraph.descriptions.items() \
                   if 'r.i.' in d or 'reinspection' in d]

def run():
    print('Reading bitbushist')
    with tbstr.indent():
        bbh = ud.enhanced.get('bbh')
    print('Determining which uids saw the reinspection')
    with tbstr.indent():
        s = pd.Series(0, index=bbh.uid.unique())
        s.loc[bbh[bbh.Tx.isin(REINSPECTION_TX)].uid.unique()] = 1
    df = pd.DataFrame({'reinspected':s}, index=s.index)
    df.index.name = 'uid'
    path = os.path.join(Paths.enhanced, 'uid_reinspection.csv')
    print('Store in {}'.format(path))
    df.to_csv(path)
