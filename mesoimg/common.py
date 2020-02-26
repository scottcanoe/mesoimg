import collections
import json
from numbers import Number
import os
import pathlib
from pathlib import Path
from pprint import PrettyPrinter
import queue
import select
import sys
import time
from typing import (Any,
                    Callable,
                    Dict,
                    List,
                    NamedTuple,
                    Tuple,
                    Union,
                    )
import h5py
import imageio
import numpy as np
import zmq


__all__ = [

    # Shares classes and constants.
    'Frame',
    'PathLike',

    # Filesystem and data I/O.
    'pathlike',
    'poll_stdin',
    'read_stdin',
    'write_stdout',
    'read_json',
    'write_json',
    'read_text',
    'write_text',
    'read_h264',
    'read_h5',
    'write_h5',
    'read_jpeg',
    'write_jpeg',
    'read_mp4',
    'write_mp4',
    'read_raw',
    'write_raw',
    'get_reader',
    'get_writer',

    # etc.
    'pi_info',
    'is_contiguous',
    'as_contiguous',
    'squeeze',
    'today',
    'repr_secs',
    'pprint',
    'pformat',

]


#------------------------------------------------------------------------------#
# Shares classes and constants


class Frame(NamedTuple):
    data: np.ndarray
    index: int
    time: Number

PathLike = Union[str, pathlib.Path]


#------------------------------------------------------------------------------#
# OS/filesystem


def pathlike(obj: Any) -> bool:
    """Determine whether an object is interpretable as a filesystem path."""
    return isinstance(obj, (str, Path))


def poll_stdin(timeout: float = 0.0) -> bool:
    """
    Returns `True` if stdin has at least one line ready to read.
    """
    return select.select([sys.stdin], [], [], timeout)[0] == [sys.stdin]


def read_stdin(timeout: float = 0.0) -> str:
    """
    Reads a line from stdin, if any. If no lines available, the empty
    string is returned.
    """
    if poll_stdin():
        return sys.stdin.readline()
    return ''


def write_stdout(chars: str, flush=True) -> None:
    sys.stdout.write(chars)
    if flush:
        sys.stdout.flush()


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


def iterframes_h264(path) -> np.ndarray:
    import cv2
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


def read_h5(path: PathLike) -> np.ndarray:
    with h5py.File(str(path), 'r') as f:
        dset = f['data']
        mov = dset[:]
    return mov


def write_h5(path: PathLike, data: np.ndarray) -> None:
    with h5py.File(str(path), 'w') as f:
        dset = f.create_dataset('data', data=data)


def read_jpeg(path: PathLike) -> np.ndarray:
    return imageio.imread(path)


def write_jpeg(path: PathLike, data: np.ndarray) -> None:
    return imageio.imwrite(path, data)


def read_mp4(path: PathLike):
    return imageio.mimread(path)


def write_mp4(path: PathLike, data: np.ndarray, fps: float = 30.0) -> None:

    imageio.mimwrite(str(path), data, fps=fps)


def read_raw(path: PathLike,
             resolution: Tuple[int, int] = (480, 480),
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

    frame_shape = [n_ypix, n_xpix] if n_channels == 1 else \
                  [n_ypix, n_xpix, n_channels]
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


def write_raw(path: PathLike, data: np.ndarray) -> None:
    with open(path, 'wb') as f:
        data.tofile(f)


def get_reader(path: PathLike, *args, **kw) -> Callable:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == 'h5':
        return read_h5
    if ext in ('jpeg', 'jpg'):
        return read_jpeg
    if ext == 'mp4':
        return read_mp4
    return read_raw



def get_writer(path: PathLike, *args, **kw) -> Callable:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ('.h5', '.hdf5'):
        return write_h5
    if ext in ('.jpeg', '.jpg'):
        return write_jpeg
    if ext == '.mp4':
        return write_mp4
    return write_raw



#------------------------------------------------------------------------------#
# etc


_pi_info = {}
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as f:
        lines = f.readlines()
    for ln in lines:
        key, val = ln.split('=')
        _pi_info[key.strip()] = val.strip()

def pi_info():
    """Infer whether this computer is a raspberry pi."""
    # Collect info about raspbian.
    return _pi_info.copy()


def is_contiguous(arr: np.ndarray) -> bool:
    return arr.flags.c_contiguous


def as_contiguous(arr: np.ndarray) -> np.ndarray:
    return arr if is_contiguous(arr) else np.ascontiguousarray(arr)


def squeeze(arr: np.ndarray) -> np.ndarray:
    """
    Call numpy.squeeze on an array only if it can be squeezed (i.e., has
    singleton dimensions).
    """
    return np.squeeze(arr) if 1 in arr.shape else arr


def today() -> str:
    """
    Returns the ISO-formatted local date (e.g., 2019-11-08).
    """
    d = time.localtime()
    return f'{d.tm_year}-{d.tm_mon:02}-{d.tm_mday:02}'


def repr_secs(secs: float) -> str:
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


_printer = PrettyPrinter()

def pprint(obj: Any) -> None:
    _printer.pprint(obj)


def pformat(obj: Any) -> str:
    return _printer.pformat(obj)

