import contextlib
import json
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
    'kill_from_procinfo',
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


def kill_from_procinfo(fname: str) -> bool:

    try:
        info = read_procinfo(fname)
    except FileNotFoundError:
        print('No procinfo for "{fname}".')
        return False

    pid = info['pid']
    if pid == os.getpid():
        print('Camera is owned by this process. Will not kill current process.')
        return False

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print('Process no longer exists. Deleting procinfo file.')
        delete_procinfo(fname)
        return False

    if proc.create_time() != info['create_time']:
        print('Create times do not match. Not killing process.')
        return False

    print(f'Killing process (pid={pid}).')
    proc.kill()
    return True

