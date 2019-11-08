import json
import os
from pathlib import Path
import shutil


CONFIGDIR = Path.home() / '.mesoimg'
APPDIR = Path(__file__).parent

DEFAULTS = {}
DEFAULTS['filedb.local'] = str(Path.home() / 'meso-db')


#------------------------------------------------------
# Utilities

def mkdir(path, exist_ok=True):
    p = Path(path)
    p.mkdir(exist_ok=exist_ok)
    return p

def read_json(path):
    with open(str(path), 'r') as f:
        return json.load()

def write_json(path, data, indent=2, **kw):

    with open(str(path), 'w') as f:
        json.dump(data, f, indent=indent, **kw)


#-----------------------------------------------------
# Setup functions


def setup_configdir():

    # Initialize directory structure.
    DIR = mkdir(CONFIGDIR)
    mkdir(DIR / 'cache')
    mkdir(DIR / 'logs')
    mkdir(DIR / 'profiles')

    # Store defaults.
    write_json(DIR / 'defaults.json', DEFAULTS)


def setup_filedb():

    # Initialize directory structure.
    DIR = Path(DEFAULTS['filedb.local'])
    mkdir(DIR)
    mkdir(DIR / 'etc')
    mkdir(DIR / 'sessions')
    mkdir(DIR / 'temp')


if __name__ == '__main__':

    setup_configdir()
    setup_filedb()



