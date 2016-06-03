import os

from tbtools.strings import fuzzy_match

_paths = {'data': os.environ['HOME'] + '/Speciale/data/' }
_paths['preconsolidated_dir_1'] = os.path.join(_paths['data'], 'NN')
_paths['preconsolidated_dir_2'] = os.path.join(_paths['data'], 'B-Efterkontrol')
_paths['raw'] = os.path.join(_paths['data'], 'exported')
_paths['raw_dbs'] = [os.path.join(os.path.join(_paths['data'], 'exported'), x)
                        for x in ('NN_1_10','NN_26','NN_CLC')]
_paths['enhanced'] = os.path.join(_paths['data'], 'enhanced')
_paths['design_matrices'] = os.path.join(_paths['data'], 'design_matrices')
_paths['cache'] = os.path.join(_paths['data'], 'cache')


class Paths:
    pass

for k,v in _paths.items():
    setattr(Paths, k, v)
    # data = os.environ['HOME'] + '/Speciale/data/'
    # preconsolidated_dir = os.path.join(data, 'NN') #/B{}.csv'.format(day)
    # raw_dir = os.path.join(data, 'exported')
    # raw_dbs = [os.path.join(raw_dir, x) for x in ('NN_1_10','NN_26','NN_CLC')]

def get(query):
    """Fuzzy search for a path label
    Returns best matching path, or raises an error.
    """
    return _paths[fuzzy_match(query, _paths)]

def get_file(approx_dir, approx_file):
    """Guesses dir based on approx_dir
    Guesses file based on dir and approx_file
    Returns the full path.
    """
    d = get(approx_dir)
    ls = os.listdir(d)
    f = fuzzy_match(approx_file, ls)
    return os.path.join(d, f)
