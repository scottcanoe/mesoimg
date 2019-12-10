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
    'read_procinfo',
    'write_procinfo',
    'delete_procinfo',
    'find_from_procinfo',
    'kill_from_procinfo',
    'Ports',
]


APPDIR = Path(__file__).parent.parent
USERDIR = Path.home() / '.mesoimg'
if not USERDIR.exists():
    USERDIR.mkdir()

for name in ('cache', 'logs', 'procinfo', 'snippets'):
    p = USERDIR / name
    if not p.exists():
        p.mkdir()


def appdir() -> Path:
    return APPDIR


def userdir() -> Path:
    return USERDIR


def read_procinfo(fname: str) -> Dict:
    path = (USERDIR / 'procinfo' / fname).with_suffix('.json')
    with open(path, 'r') as f:
        return json.load(f)


def write_procinfo(fname: str) -> None:

    proc = psutil.Process()
    info = {'pid' : proc.pid,
            'create_time' : proc.create_time(),
            'name' : proc.name()}
    path = (USERDIR / 'procinfo' / fname).with_suffix('.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)


def delete_procinfo(fname: str) -> Dict:
    path = (USERDIR / 'procinfo' / fname).with_suffix('.json')
    if path.exists():
        path.unlink()


def find_from_procinfo(fname: str) -> Optional[psutil.Process]:

    try:
        info = read_procinfo(fname)
    except FileNotFoundError:
        return

    try:
        proc = psutil.Process(info['pid'])
    except psutil.NoSuchProcess:
        delete_procinfo(fname)
        return

    if proc.create_time() != info['create_time']:
        return

    return proc


def kill_from_procinfo(fname: str) -> None:

    proc = find_from_procinfo(fname)
    if proc is None:
        logging.info(f"No existing process found matching info for '{fname}'")
        return

    logging.info(f'Killing process (pid={proc.pid}).')
    proc.kill()



@unique
class Ports(IntEnum):
    """
    Enum for holding port numbers.
    """
    COMMAND    = 7000   # req/rep (rep on server side)
    CONSOLE    = 7001   # req/rep (rep on server side)
    FRAME_PUB  = 7004   # pub/sub
    STATUS_PUB = 7005   # pub/sub


