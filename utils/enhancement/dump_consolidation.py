import os

from .. import data as ud

import tbtools.strings as tbstr

def run():
    print('Consolidating everything')
    with tbstr.indent():
        df = ud.consolidation.get(settings='enhanced', progress_bar=True)
    path = os.path.join(ud.paths.Paths.enhanced, 'consolidated.csv')
    print('Storing results in {}'.format(path))
    df.to_csv(path)
