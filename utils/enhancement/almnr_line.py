import os

from .. import data as ud

import tbtools.strings as tbstr

def run():
    print('Writing alarm number line associations to disk')
    alm = ud.raw.get('almhist')
    alm = alm.reset_index()[['AlmNr', 'TB1']].drop_duplicates()
    alm['L1_hard'] = alm.TB1.str.contains('L1').fillna(False)
    alm['L2_hard'] = alm.TB1.str.contains('L2').fillna(False)
    alm['L1_soft'] = ~alm['L2_hard'] | alm['L1_hard']
    alm['L2_soft'] = ~alm['L1_hard'] | alm['L2_hard']

    path = os.path.join(ud.paths.Paths.design_matrices, 'almnr_lines')
    alm.to_pickle(path)
    print('Storing results in {}'.format(path))