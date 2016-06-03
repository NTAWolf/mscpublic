import random
from functools import lru_cache
import tbtools.strings as tbstr
import os
import pandas as pd

# Upper-case chars for use in rand_char
CHARS = ''.join([chr(v) for v in range(65, 91)])

def rand_char():
    return random.choice(CHARS)

class TempFile:

    def __init__(self, dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as ose:
            # Folder already exists.
            pass

        self.path = TempFile.generate_path(dir_path)

    @staticmethod
    def generate_path(dir_path):
        candidate = rand_char()
        while TempFile.is_in_dir(candidate, dir_path):
            candidate += rand_char()
        return os.path.join(dir_path, candidate)

    @staticmethod
    def is_in_dir(file_name, dir_path, ignore_ending=True, ignore_case=True):
        p = TempFile.normalize(file_name, ignore_ending, ignore_case)
        files = os.listdir(dir_path)
        for f in files:
            if p == TempFile.normalize(f, ignore_ending, ignore_case):
                return True
        return False

    @staticmethod
    def normalize(file_name, ignore_ending, ignore_case):
        if ignore_ending:
            file_name = file_name.rsplit('.', 1)[0]
        if ignore_case:
            file_name = file_name.lower()
        return file_name

    def update_ending(self):
        """The dbc program apparently appends .csv to
        the target path. This method automates the detection
        and update of tmp file path to recognize that.
        """
        dir_path, p = os.path.split(self.path)
        files = os.listdir(dir_path)
        for f in files:
            try:
                base, end = f.split('.')
            except ValueError:
                continue
            if p == base:
                self.path += '.' + end
                return

    def __enter__(self):
        self.handle = open(self.path, self.mode) 
        return self.handle

    def __exit__(self, _, __, ___):
        self.handle.close()

    def delete(self):
        if os.path.isfile(self.path):
            os.remove(self.path)

    def open(self, mode):
        self.mode = mode
        return self



class DataFrameBase:
    """A pandas-DataFrame-like collection
    of pandas DataFrames, for data collected 
    in separate .csv files in a single directory.
    
    It is read-only with respect to the source data.
    
    Also, guesses the best matching directory and
    DataFrame names given slightly incorrect input.
    """
    
    def __init__(self, path, name=None, 
                 global_settings=None, 
                 table_settings=None):
        """path is a string representing the path to the 
        directory which is to be represented as a database
        
        If name is supplied, name is a string that can be
        matched to a directory name in path. Then that
        directory will be the source of the database.       

        global_settings is a dict. It is used when retrieving
            the DB tables, in pandas.read_csv(**global_settings)

        table_settings is a dict where the keys are 
            table names. The values are also dict, to be 
            passed as such: pandas.read_csv(**table_settings).
        """
        if name is not None:
            name = tbstr.fuzzy_match(name, os.listdir(path))
            path = os.path.join(path, name)
        self.path = path
        self.name = os.path.split(path)[1] # Last part of path
        
        self.global_settings = global_settings or dict()
        self.table_settings = table_settings or dict()
        
        tables = os.listdir(path)
        tables = [t[:-4] for t in tables if t.endswith('.csv')]
        self.tables = tables
    
    def __getitem__(self, key):
        if not key in self.tables:
            key = tbstr.fuzzy_match(key, self.tables)
        return self._get_table(key)
    
    def __getattr__(self, key):
        return self[key]
        
    @lru_cache(maxsize=32)
    def _get_table(self, table):
        kwargs = self.global_settings.copy()
        kwargs.update(self.table_settings.get(table, {}))
        table += '.csv'
        path = os.path.join(self.path, table)
        return pd.read_csv(path, **kwargs)
        