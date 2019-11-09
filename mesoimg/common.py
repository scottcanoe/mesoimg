import os
from pathlib import Path
import time
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np


__all__ = [

    # Constants
    'ArrayTransform',
    'PathLike',
    'uint8',
    
    # OS/filesystem.
    'pi_info',
    'pathlike',
    'remove',
    'read_json',
    'write_json',
    'read_text',
    'write_text',

    # etc.
    'squeeze',
    'Clock',
    'Timer',
]


# Define useful constants.
ArrayTransform = Callable[[np.ndarray], np.ndarray]
PathLike = Union[str, bytes, Path]
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


def pathlike(obj: Any) -> bool:
    """Determine whether an object is interpretable as a filesystem path."""
    try:
        os.fspath(obj)
        return True
    except:
        return False
        

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
    
    
#from collections import UserDict
#class DotDict:
    
    
#    def __init__(self, **kw):
#        self.data = kw.copy()
#        self.x = 0

 #   def __getattr__(self, key):
  #      print('getattr failed.')
 

#d = Dict(a=1,  b='BB')

    
