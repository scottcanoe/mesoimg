import os
from pathlib import Path





def setup_userdir():

    userdir = Path.home() / '.mesoimg'
    userdir.mkdir(exist_ok=True)

    for name in ('logs', 'procinfo'):
        p = userdir / name
        p.mkdir(exist_ok=True)



if __name__ == '__main__':

    setup_userdir()