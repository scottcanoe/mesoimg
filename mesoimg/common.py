import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple, Union
import urllib.parse
import numpy as np


__all__ = [

    # Constants/type hints.
    'ArrayTransform',
    'PathLike',
    'URLLike',
    'uint8',
    
    # OS/filesystem.
    'pi_info',
    'remove',
    'read_json',
    'write_json',
    'read_text',
    'write_text',
    
    # URL/path handling.
    'pathlike',
    'urllike',
    'parse_url',

    # etc.
    'squeeze',
]


# Define useful constants.
ArrayTransform = Callable[[np.ndarray], np.ndarray]
PathLike = Union[str, Path]
URL = Union[str, bytes, Path, urllib.parse.ParseResult]
uint8 = np.dtype('>u1')

# Collect info about raspbian.
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as f:
        lines = f.readlines()
    _PI_INFO = {}
    for ln in lines:
        key, val = ln.split('=')
        _PI_INFO[key.strip()] = val.strip()
else:
    _PI_INFO = None



"""
OS/filesystem utilities.
"""


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
        

def write_json(path: PathLike, data: dict, indent: int=2, **kw) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, **kw)


def read_text(path: PathLike) -> str:
    with open(path, 'r') as f:
        return f.read()


def write_text(path: PathLike, text: str) -> None:
    with open(path, 'w') as f:
        f.write(text)
       


        
# URL and path handling.

def pathlike(obj: Any) -> bool:
    """Determine whether an object is interpretable as a filesystem path."""
    try:
        return isinstance(obj, (str, Path))
    except:
        return False
        

def urllike(obj: Any) -> bool:
    """Determine whether an object is interpretable as a filesystem path."""
    if pathlike(obj) or isinstance(obj, (ParseResult, URL)):
        return True
    return False        



class DictView:

    def __init__(self, data: Mapping):
        self._data = data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def pop(self, key):
        raise TypeError('Cannot modify DictView')
                
    def update(self, other: Mapping):
        raise TypeError('Cannot modify DictView')

    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, val):
        raise TypeError('Cannot modify DictView')
        
    def __delitem__(self, key):
        raise TypeError('Cannot modify DictView')
    
    def __contains__(self, key):
        return key in self.keys()
        
    def __len__(self):
        return len(self._data)


    
    
            
def urlparse(url: PathLike,
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
                

# etc.        

def squeeze(arr: np.ndarray) -> np.ndarray:
    """
    Call numpy.squeeze on an array only if it can be squeezed (i.e., has singleton
    dimensions).
    """
    return np.squeeze(arr) if 1 in arr.shape else arr
    
        

        
        
def load_raw(path: PathLike,
             resolution: Tuple[int, int] = (640, 480),
             n_channels: int = 3) -> np.ndarray:
    """
    Load raw (unencoded) RGB-formatted video or image data, returning it 
    with indices ordered as [t, y, x, c], where 't' and/or 'c' axes may be
    absent depending on whether the data is a single image (no time axis) or
    single-channel (no channel axis).
    
    """

    
    data = np.fromfile(path, dtype=uint8)    
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
    mov = np.zeros(out_shape, dtype=uint8)
    for i in range(n_frames):
        frame_data = data[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        mov[i] = frame_data.reshape(frame_shape)
    return mov        
    








