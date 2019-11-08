import os
from pathlib import Path
import time
from typing import Any, Tuple, Union
import numpy as np


__all__ = [
    'PathLike',
    'uint8',
    'clear_path',
    'is_rpi',
    'load_raw',
    'squeeze',
    'Clock',
]


_IS_RPI = 'arm' in os.uname().machine
PathLike = Union[str, bytes, Path]
uint8 = np.dtype('>u1')


def clear_path(path: PathLike) -> Path:
    """Remove a file if it exists, returning a valid writeable path."""
    path = Path(path)
    if path.exists():
        path.unlink()
    return path


def is_rpi():
    """Infer whether this computer is a raspberry pi."""
    return _IS_RPI

        
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

        
def pathlike(obj: Any) -> bool:
    """Determine whether an object is path-like."""
    try:
        os.fsencode(obj)
        return True
    except:
        return False


def squeeze(obj):
    """Squeeze an array only if it has singleton dimensions."""
    return np.squeeze(obj) if 1 in obj.shape else obj
    
        
class Clock:
    
    def __init__(self, start: bool = False):
        self.reset()
        if start:
            self.start()

    def reset(self):
        self._t_start = None
        self._t_stop = None
        
    def start(self) -> float:
        self._t_start = time.perf_counter()
        return 0

    def time(self) -> float:
        return time.perf_counter() - self._t_start
    
    def stop(self) -> float:
        self._t_stop = time.perf_counter()
        return self._t_stop - self._t_start    