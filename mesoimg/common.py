import os
from pathlib import Path
import time
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union
import urllib
from urllib.parse import urlparse, ParseResult
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
    'Clock',
    'Timer',
        
]


# Define useful constants.
ArrayTransform = Callable[[np.ndarray], np.ndarray]
PathLike = Union[str, bytes, Path]
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
        os.fspath(obj)
        return True
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


import re


from collections import namedtuple


    

    
class URL:
        
    
    _fields = ('scheme',
               'netloc',
               'path',
               'params',
               'query',
               'fragment')

    _attrs = ('hostname',
              'password',
              'port')
              
    
    
    def __new__(cls,
                url: URLLike,
                *,
                copy: bool = False,
                **kw):

        scheme = kw.pop('scheme', None)
        flags = kw

        # Parse path-like input.
        if isinstance(url, (bytes, str, Path)):
            res = urlparse(url)

        # Store parse results.
        elif isinstance(url, ParseResult):
            res = url

        # Return a copy of the template URL.
        elif isinstance(url, URL):
            if copy:
            
                # Copy over field and attribute values to a new instance.
                instance = object.__new__(cls)
                for group in (cls._fields, cls._attrs):
                    for name in group:
                        setattr(instance, '_' + name, getattr(url, name))                      
                
                return instance

            return url

        else:
            raise ValueError("{} is not a valid URL.".format(url))
        
        
        # Finish parsing the parse result.
        # If a scheme was supplied, override what was found by urlparse.
            
            
        if scheme is not None:
            res = res._replace(scheme=scheme)
        
        if 'scheme' in kw:
            scheme = kw.pop('scheme')
            
            
             
    
    
            
def parse_url(url: URLLike,
              copy: bool = True,
              READONLY: bool = False) -> ParseResult:
    
    if 'scheme' in kw:
        scheme = kw['scheme']
        force_scheme = True
    else:
        force_scheme = False
        
    if isinstance(url, URL):
        return url
    
    if isinstance(url, (bytes, str, Path)):
        res = urlparse(os.fsdecode(url))
    elif isinstance(url, ParseResult):
        res = url
    elif isinstance(url, URL):
        if copy:
            return url.copy()
        return url

    res = url if isinstance(url, ParseResult) else urlparse(os.fsdecode(url))
    return res
                



# etc.        

def squeeze(arr: np.ndarray) -> np.ndarray:
    """
    Call numpy.squeeze on an array only if it can be squeezed (i.e., has singleton
    dimensions).
    """
    return np.squeeze(arr) if 1 in arr.shape else arr
    
        
class Clock:

    """
    Class to assist in timing intervals.
    """
    
    def __init__(self,
                 fn: Callable = time.perf_counter,    
                 start: bool = False):

        self._fn = fn
        self.reset()
        if start:
            self.start()

                    
    def start(self) -> float:
        if self._running:
            raise RunTimeError("Clock already started.")            
        self._t_start = self._fn()
        self._running = True
        return 0
        
    def time(self) -> float:
        if not self._running:
            raise RunTimeError('Clock is stopped.')
        return self._fn() - self._t_start

    def stop(self) -> float:
        self._t_stop = self.time()
        self._running = False
        return self._t_stop

    def reset(self) -> None:
        self._t_start = None
        self._t_stop = None
        self._running = False
        
    
class Timer:

    """
    Class to assist in timing intervals.
    """
    
    def __init__(self,
                 fn: Callable = time.perf_counter,
                 start: bool = False,
                 ID: Any = None,
                 verbose: bool = False):

        self._fn = fn
        self._ID = ID
        self._verbose = verbose
        self.reset()
        if start:
            self.start()

    @property
    def ID(self) -> Any:
        return self._ID
        
    @property
    def timestamps(self):
        return self._timestamps
        
    def reset(self, start: bool = False) -> Optional[float]:
        self._t_start = None
        self._t_stop = None
        self._running = False
        self._timestamps = []
        if start:
            self.start()  
                    
                    
    def start(self) -> float:
        if self._running:
            raise RunTimeError("Clock already started.")      
        self._t_start = self._fn()
        self._timestamps = [0]
        self._running = True
        return 0
        
        
    def tic(self) -> float:
        if not self._running:
            raise RunTimeError('Clock is stopped.')
        t = self._fn() - self._t_start
        self._timestamps.append(t)
        return t


    def stop(self, verbose: Optional[bool] = None) -> float:
        self._t_stop = self.tic()
        self._running = False

        verbose = self._verbose if verbose is None else verbose
        if verbose:
            self.print_summary()
        
        return self._t_stop


    def print_summary(self) -> None:
        s  = f'<Timer (ID={self.ID}): '
        s += f'elapsed={self._t_stop}>'
        print(s, flush=True)
        
        
        
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
    








