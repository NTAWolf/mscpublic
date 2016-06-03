import os
import subprocess
from util import pause, TempFile
import codecs

class MSSQLExporter:

    def __init__(self, dbnames, target_dir_names, target_dir):
        self.dbnames = dbnames
        self.target_dir_names = target_dir_names
        self.target_dir = target_dir
        self.tmp_dir = os.path.join(target_dir, 'tmp')

    def convert_dbs(self):
        for dbname, dirname in zip(self.dbnames, self.target_dir_names):
            db_dir = os.path.join(self.target_dir, dirname)
            try:
                os.makedirs(db_dir)
            except OSError as ose:
                print("Directory at {} already exists. Proceeding.".format(db_dir))
            else:
                print("Directory created at {} ... Proceeding.".format(db_dir))

            self.convert_db(dbname, db_dir)
            
    def convert_db(self, dbname, target_dir):
        tnames = self.get_table_names(dbname)
        print("Converting {} ({} tables)".format(dbname, len(tnames)))
        for tablename in tnames:
            self.convert_table(dbname, tablename, target_dir)

    def convert_table(self, dbname, tablename, target_dir):
        print("\tConverting {}.{}".format(dbname, tablename))
        cols = self.get_column_names(dbname, tablename)
        cols = ';'.join(cols) + '\n'
        
        targetpath = os.path.join(target_dir, tablename + '.csv')
        tmpfile = TempFile(self.tmp_dir)

        self.write_data(dbname, tablename, tmpfile.path)
        # The underlying dbc tends to append an ending
        tmpfile.update_ending() 

        i = 0
        with open(targetpath, 'w') as f:
            f.write(cols)
            with tmpfile.open('rb') as t:
                # Decode as utf-16
                # Returns (decoded_text, n_chars)
                text = codecs.utf_16_decode(t.read())[0]
                # replace NULL with nothing
                text = text.replace('\x00', '')
                # convert to *nix line endings
                text = text.replace('\r\n', '\n')

                f.write(text)
        tmpfile.delete()

    def get_table_names(self, dbname):
        """Returns table names as a python list of strings
        """

        tmpfile = TempFile(self.tmp_dir)
        
        command = ("BCP \"USE {dbname}; SELECT name FROM sys.tables;\" "
                   "queryout {target} -c -T").\
                       format(dbname=dbname, target=tmpfile.path)

        subprocess.run(command)
        tmpfile.update_ending()
            
        with tmpfile.open('r') as f:
            res = f.read()
            
        tmpfile.delete()

        return res.strip().split()
        
    def get_column_names(self, dbname, tablename):
        """Returns column names as a python list of strings
        """
        
        tmpfile = TempFile(self.tmp_dir)
        
        command = ("BCP \"DECLARE @colnames VARCHAR(max);"
            "SELECT @colnames = COALESCE(@colnames + ',', '') + column_name "
            "from {dbname}.INFORMATION_SCHEMA.COLUMNS "
            "where TABLE_NAME='{tablename}'; "
            "select @colnames;\" "
            "queryout {target} -c -T").\
                    format(dbname=dbname, tablename=tablename, target=tmpfile.path)
        subprocess.run(command)
        
        with tmpfile.open('r') as f:
            res = f.read()
            
        tmpfile.delete()

        return res.strip().split(',')

    def write_data(self, dbname, tablename, targetpath):
        # command = 'bcp {dbname}..{tablename} out {targetpath}.csv -T -c -r"\\n" -t ;'.\
        command = "bcp {dbname}..{tablename} out {targetpath} -T  -w -t ;".\
            format(dbname=dbname, tablename=tablename, targetpath=targetpath)
        subprocess.run(command)



target_folder = "E:\\data\\exported\\"

dbnames = ("NN_NPP_1_10", "NN_NPP_26", "NN_NPP_CLC")
target_dir_names = ("NN_1_10", "NN_26", "NN_CLC")


def run():
    e = MSSQLExporter(dbnames, target_dir_names, target_folder)
    e.convert_dbs()
