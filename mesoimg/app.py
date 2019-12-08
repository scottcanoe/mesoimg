import contextlib
import os
from pathlib import Path
import time
from typing import Any, Optional, Union
import subprocess as sp
import psutil


__all__ = [
    'appdir',
    'userdir',
    'read_snippet',
    'write_snippet',
    'delete_snippet',
    'kill_zombied_camera',
]


APPDIR = Path(__file__).parent.parent
USERDIR = Path.home() / '.mesoimg'
if not USERDIR.exists():
    USERDIR.mkdir()

for name in ('cache', 'logs', 'snippets'):
    p = USERDIR / name
    if not p.exists():
        p.mkdir()


def appdir() -> Path:
    return APPDIR


def userdir() -> Path:
    return USERDIR


def read_snippet(name: str, *default) -> Any:
    path = USERDIR / 'snippets' / name
    if not path.exists() and default:
        return default[0]
    return path.read_text()


def write_snippet(name: str, text: str) -> None:
    path = USERDIR / 'snippets' / name
    path.write_text(text)


def delete_snippet(name: str) -> None:
    path = USERDIR / 'snippets' / name
    if path.exists():
        path.unlink()


def kill_zombied_camera():

    fname = 'picamera.pid'

    pid = int(read_snippet(fname, 0))
    if pid == 0 or pid == os.getpid():
        return

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        delete_snippet(fname)

    if proc.name() != 'python':
        return

    print(f'Killing process that last created a PiCamera instance (pid={pid}).')
    proc.kill()
    time.sleep(1)
    delete_snippet(fname)

