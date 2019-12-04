from collections import namedtuple
import os
import pathlib
from pathlib import Path
import select
import sys
from threading import Event
import time
from typing import (Any,
                    Callable,
                    ItemsView,
                    Iterable,
                    KeysView,
                    Mapping,
                    Optional,
                    Tuple,
                    Union,
                    ValuesView)
import urllib.parse
import numpy as np


__all__ = [

    # Constants/type hints
    'PathLike',
    
    # OS/filesystem
    'pi_info',
    'remove',
    'read_json',
    'write_json',
    'read_text',
    'write_text',
    'read_raw',
    'read_h264',
        
    # URL/path handling
    'pathlike',
    'urlparse',

    # User interaction
    'stdin_ready',
    'read_stdin',
        
    # Networking
    'Ports',


    # etc.
    'squeeze',
    'today',
    'repr_secs',

    
]


# Define useful constants.
PathLike = Union[str, pathlib.Path]


"""
OS/filesystem utilities.
"""

# Collect info about raspbian.
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as f:
        lines = f.readlines()
    _PI_INFO = {}
    for ln in lines:
        key, val = ln.split('=')
        _PI_INFO[key.strip()] = val.strip()


def pi_info():
    """Infer whether this computer is a raspberry pi."""
    return _PI_INFO




def remove(path: PathLike) -> Path:
    """
    Equivalent to os.remove without raising an error if the file does
    not exist.
    """
    path = Path(path)
    if path.exists():
        path.unlink()
    return path



def read_json(path: PathLike) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
        

def write_json(path: PathLike, data: dict, indent: int = 2, **kw) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, **kw)


def read_text(path: PathLike) -> str:
    with open(path, 'r') as f:
        return f.read()


def write_text(path: PathLike, text: str) -> None:
    with open(path, 'w') as f:
        f.write(text)
       

def read_raw(path: PathLike,
             resolution: Tuple[int, int] = (640, 480),
             n_channels: int = 3,
             ) -> np.ndarray:
    """
    Load raw (unencoded) RGB-formatted video or image data, returning it 
    with indices ordered as [t, y, x, c], where 't' and/or 'c' axes may be
    absent depending on whether the data is a single image (no time axis) or
    single-channel (no channel axis).
    
    """
    
    data = np.fromfile(str(path), dtype=np.uint8)
    n_bytes = len(data)
    n_xpix, n_ypix = resolution
    bytes_per_frame = n_xpix * n_ypix * n_channels
    n_frames, remainder = np.divmod(n_bytes, bytes_per_frame)
    if n_frames < 1:
        raise IOError("No frames found in file '{}'.".format(path))        
    if remainder != 0:
        raise IOError("Partial frames encountered in file '{}'.".format(path))

    frame_shape = [n_ypix, n_xpix] if n_channels == 1 else [n_ypix, n_xpix, n_channels]
    out_shape = frame_shape if n_frames == 1 else [n_frames] + frame_shape
    
    # Handle single frame.
    if n_frames == 1:
        return data.reshape(frame_shape)
            
    # Handle multiple frames (video).
    mov = np.zeros(out_shape, dtype=np.uint8)
    for i in range(n_frames):
        frame_data = data[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        mov[i] = frame_data.reshape(frame_shape)
    return mov        
    

def iterframes_h264(path) -> np.ndarray:
    
    cap = cv2.VideoCapture(str(path))
    try:
        while cap.isOpened():
            ret, im = cap.read()
            if not ret:
                break
            im = im[:, :, (2, 1 , 0)]
            yield im

    finally:
        cap.release()


def read_h264(path: PathLike) -> np.ndarray:
    return np.array(list(iterframes_h264(path)))

        


# URL and path handling.


def pathlike(obj: Any) -> bool:
    """Determine whether an object is interpretable as a filesystem path."""
    return isinstance(obj, (str, Path))        

            
def urlparse(url: Union[PathLike, urllib.parse.ParseResult],
             scheme: str = '',
             ) -> urllib.parse.ParseResult:
    """
    Like urllib.parse.urlparse but capable of handling pathlib paths and 
    parse results.
    """
    if isinstance(url, urllib.parse.ParseResult):
        url = url._replace(scheme=scheme) if scheme else url
        return url
    
    url = str(url) if isinstance(url, (bytes, Path)) else url
    if not isinstance(url, str):
        raise ValueError(f"invalid argument '{url}' for urlparse.")
    
    res = urllib.parse.urlparse(url)
    return res
                


# User interaction.

def stdin_ready(timeout: float = 0.0) -> bool:
    """
    Determine whether stdin has at least one line available to read.
    """
    return select.select([sys.stdin], [], [], timeout)[0] == []


def read_stdin(timeout: float = 0.0) -> str:
    """
    Reads a line from stdin, if any. If no lines available, the empty
    string is returned.
    """
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.readline()
    return ''


# Networking

from enum import IntEnum

class Ports(IntEnum):
    COMMAND = 7000
    FRAME   = 7001
    STATUS  = 7002
    PREVIEW = 7003


# etc.


def squeeze(arr: np.ndarray) -> np.ndarray:
    """
    Call numpy.squeeze on an array only if it can be squeezed (i.e., has singleton
    dimensions).
    """
    return np.squeeze(arr) if 1 in arr.shape else arr
    
        
def today() -> str:
    """
    Returns the ISO-formatted local date (e.g., 2019-11-08).
    """
    d = time.localtime()
    return f'{d.tm_year}-{d.tm_mon:02}-{d.tm_mday:02}'
       

def repr_secs(secs):
    """
    """

        
    sign, secs = np.sign(secs), np.abs(secs)
    if secs >= 1:
        return sign * secs, 'sec.'
    if secs >= 1e-3:
        return sign * secs * 1e3, 'msec.'
    elif secs >= 1e-6:
        return sign * secs * 1e6, 'usec.' 
    else:
        return sign * secs * 1e9, 'nsec'
        


