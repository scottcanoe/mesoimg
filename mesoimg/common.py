from collections import namedtuple
from enum import IntEnum
import json
import logging
import os
import pathlib
from pathlib import Path
from pprint import PrettyPrinter
import select
import sys
from threading import Event
import time
from typing import (Any,
                    Callable,
                    NamedTuple,
                    Tuple,
                    Union,
                    )
import urllib.parse
import h5py
import numpy as np
import zmq


__all__ = [

    # Networking
    'ZMQContext',
    'ZMQSocket',
    'ZMQPoller',
    'Ports',
    'Frame',
    'send_array',
    'recv_array',
    'pub_array',
    'sub_array',
    'send_frame',
    'recv_frame',
    'pub_frame',
    'sub_frame',
    'poll_stdin',
    'read_stdin',

    # Filesystem and data I/O.
    'PathLike',
    'pathlike',
    'pi_info',
    'remove',
    'read_json',
    'write_json',
    'read_text',
    'write_text',
    'read_raw',
    'read_h264',
    'read_h5',
    'write_h5',
    'write_mp4',

    # etc.
    'is_contiguous',
    'as_contiguous',
    'squeeze',
    'today',
    'repr_secs',
    'pprint',
    'pformat',
]


#------------------------------------------------------------------------------#
# Networking


ZMQContext = zmq.sugar.context.Context
ZMQSocket = zmq.sugar.socket.Socket
ZMQPoller = zmq.sugar.Poller


class Ports(IntEnum):
    """
    Enum for holding port numbers.
    """
    COMMAND    = 7000
    FRAME_PUB  = 7001
    STATUS_PUB = 7002


class Frame(NamedTuple):
    """
    Named tuple that encapsulates an imaging frame along
    frame number and a timestamp.
    """
    data: np.ndarray
    index: int
    timestamp: float


def send_array(socket: ZMQSocket,
               arr: np.ndarray,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> None:
    """
    Send a `Frame` object over a zmq socket.
    """
    md = {'shape' : arr.shape, 'dtype' : str(arr.dtype)}
    socket.send_json(md, flags | zmq.SNDMORE)
    socket.send(arr, flags, copy, track)


def recv_array(socket: ZMQSocket,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> np.ndarray:
    """
    Receive an ndarray over a zmq socket.
    """
    md = socket.recv_json(flags)
    buf  = memoryview(socket.recv(flags, copy, track))
    return np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])


def pub_array(socket: ZMQSocket,
              arr: np.ndarray,
              topic: str,
              flags: int = 0,
              copy: bool = True,
              track: bool = False,
              ) -> None:
    """
    Send a `Frame` object over a zmq socket.
    """

    socket.send_string(topic, flags | zmq.SNDMORE)
    send_array(socket, arr, flags, copy, track)


def sub_array(socket: ZMQSocket,
              flags: int = 0,
              copy: bool = True,
              track: bool = False,
              ) -> np.ndarray:
    """
    Receive an ndarray over a zmq socket.
    """
    topic = socket.recv_string(flags)
    md = socket.recv_json(flags)
    buf  = memoryview(socket.recv(flags, copy, track))
    return np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])


def send_frame(socket: ZMQSocket,
               frame: Frame,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> None:
    """
    Send a `Frame` object over a zmq socket.
    """
    md = {'shape': frame.data.shape,
          'dtype': str(frame.data.dtype),
          'index': frame.index,
          'timestamp' : frame.timestamp}
    socket.send_json(md, flags | zmq.SNDMORE)
    socket.send(frame.data, flags, copy, track)


def recv_frame(socket: ZMQSocket,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> Frame:
    """
    Receive a `Frame` object over a zmq socket.
    """

    md = socket.recv_json(flags)
    buf = memoryview(socket.recv(flags, copy, track))
    data = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])
    return Frame(data, index=md['index'], timestamp=md['timestamp'])


def pub_frame(socket: ZMQSocket,
              frame: Frame,
              topic: str,
              flags: int = 0,
              copy: bool = True,
              track: bool = False,
              ) -> None:
    """
    Publish a `Frame` object.
    """
    socket.send_string(topic, flags | zmq.SNDMORE)
    send_frame(socket, frame, flags, copy, track)


def sub_frame(socket: ZMQSocket,
              flags: int = 0,
              copy: bool = True,
              track: bool = False,
              ) -> Frame:
    """
    Subscribe/recv a `Frame` object.
    """
    topic = socket.recv_string(flags)
    return recv_frame(socket, flags, copy, track)


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
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.readline()
    return ''


#------------------------------------------------------------------------------#
# OS/filesystem

PathLike = Union[str, pathlib.Path]

def pathlike(obj: Any) -> bool:
    """Determine whether an object is interpretable as a filesystem path."""
    return isinstance(obj, (str, Path))


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

def write_h5(path: PathLike, arr: np.ndarray) -> None:

    with h5py.File(str(path), 'w') as f:
        dset = f.create_dataset('data', data=arr)


def write_mp4(path: PathLike, mov: np.ndarray, fps: float=30.0) -> None:
    import imageio
    imageio.mimwrite(str(path), mov, fps=fps)


#------------------------------------------------------------------------------#
# etc

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



printer = PrettyPrinter()

def pprint(obj: Any) -> None:
    printer.pprint(obj)


def pformat(obj: Any) -> str:
    return printer.pformat(obj)
