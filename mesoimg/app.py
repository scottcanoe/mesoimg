import contextlib
from enum import IntEnum, unique
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import subprocess as sp
import psutil


__all__ = [
    'appdir',
    'userdir',
    'logdir',
    'save_procinfo',
    'load_procinfo',
    'from_procinfo',
    'Ports',
]



APPDIR = Path(__file__).parent.parent
USERDIR = Path.home() / '.mesoimg'


def appdir() -> Path:
    return APPDIR


def userdir() -> Path:
    return USERDIR


def logdir() -> Path:
    return USERDIR / 'logs'



"""
Process tracking utilities.

"""


def save_procinfo(name: str, pid: Optional[int] = None) -> None:
    """
    Save process info to a file in case we need to terminate
    it later when we need to recover its resources.
    """
    pid = os.getpid() if pid is None else pid
    proc = psutil.Process(pid)
    info = {'pid' : proc.pid,
            'create_time' : proc.create_time(),
            'name' : proc.name()}
    path = (USERDIR / 'procinfo' / name).with_suffix('.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)


def load_procinfo(name: str) -> Dict:
    """
    Read stored process info.
    """
    path = (USERDIR / 'procinfo' / name).with_suffix('.json')
    with open(path, 'r') as f:
        return json.load(f)



def from_procinfo(name: str) -> Optional[psutil.Process]:

    try:
        info = load_procinfo(name)
    except FileNotFoundError:
        return

    try:
        proc = psutil.Process(info['pid'])
    except psutil.NoSuchProcess:
        path = (USERDIR / 'procinfo' / name).with_suffix('.json')
        path.unlink()
        return

    if proc.create_time() != info['create_time']:
        return

    return proc



@unique
class Ports(IntEnum):
    """
    Enum for holding port numbers.
    """
    COMMAND    = 7000   # req/rep (rep on server side)
    CONSOLE    = 7001   # req/rep (rep on server side)

    CAM_PAIR   = 7010
    CAM_FRAME  = 7011
    CAM_META   = 7012


"""
Setup logging.
"""

logfile = logdir() / 'log.txt'
logging.basicConfig(filename=logfile,
                    #filemode='w',
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)





