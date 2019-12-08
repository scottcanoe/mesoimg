import contextlib
import os
from pathlib import Path
from typing import Any, Optional, Union
import subprocess as sp

__all__ = [
    'appdir',
    'userdir',
    'init_userdir',
    'read_snippet',
    'write_snippet',
    'init_userdir',
]


APPDIR = Path(__file__).parent.parent
USERDIR = Path.home() / '.mesoimg'


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






def init_userdir():
    if not USERDIR.exists():
        USERDIR.mkdir()

    for name in ('cache', 'logs', 'snippets'):
        p = USERDIR / name
        if not p.exists():
            p.mkdir()

init_userdir()