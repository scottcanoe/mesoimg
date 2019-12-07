from collections import namedtuple
from enum import IntEnum
import json
import logging
import multiprocessing
import os
import pathlib
from pathlib import Path
from pprint import PrettyPrinter
import select
import sys
import threading
import time
from typing import (Any,
                    Callable,
                    Dict,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union,
                    )
import queue
import h5py
import numpy as np
import zmq


__all__ = [

    # Networking
    'Ports',
    'Frame',
    'send_bytes',
    'recv_bytes',
    'send_json',
    'recv_json',
    'send_string',
    'recv_string',
    'send_pyobj',
    'recv_pyobj',
    'send_array',
    'recv_array',
    'send_frame',
    'recv_frame',
    'poll_stdin',
    'read_stdin',

    # Threading, multiprocessing, etc.
    'clear_queue',
    'push_queue',
    'read_queue',

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


def send_bytes(socket: zmq.Socket, data: bytes, **kw) -> None:
    """
    Send bytes.
    """
    socket.send(data, **kw)


def recv_bytes(socket: zmq.Socket, **kw) -> bytes:
    """
    Receive bytes.
    """
    return socket.recv(**kw)


def send_json(socket: zmq.Socket, data: dict, **kw) -> None:
    """
    Send a dictionary.
    """
    socket.send_json(data, **kw)


def recv_json(socket: zmq.Socket, **kw) -> Dict:
    """
    Receive a dictionary.
    """
    return socket.recv_json(**kw)


def send_string(socket: zmq.Socket, data: str, **kw) -> None:
    """
    Send a string.
    """
    socket.send_string(data, **kw)


def recv_string(socket: zmq.Socket, **kw) -> str:
    """
    Receive a string.
    """
    return socket.recv_string(**kw)


def send_pyobj(socket: zmq.Socket, data: Any, **kw) -> None:
    """
    Send python object.
    """
    socket.send_pyobj(data, **kw)


def recv_pyobj(socket: zmq.Socket, **kw) -> Any:
    """
    Receive python object.
    """
    return socket.recv_pyobj(**kw)


def send_array(socket: zmq.Socket,
               data: np.ndarray,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> None:
    """
    Send a ndarray.
    """

    md = {'shape' : data.shape, 'dtype' : str(data.dtype)}
    socket.send_json(md, flags | zmq.SNDMORE)
    socket.send(data, flags, copy, track)


def recv_array(socket: zmq.Socket,
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


def send_frame(socket: zmq.Socket,
               data: Frame,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> None:
    """
    Send a `Frame` object over a zmq socket.
    """
    md = {'shape': data.data.shape,
          'dtype': str(data.data.dtype),
          'index': data.index,
          'timestamp' : data.timestamp}
    socket.send_json(md, flags | zmq.SNDMORE)
    socket.send(data.data, flags, copy, track)


def recv_frame(socket: zmq.Socket,
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
    return Frame(data=data, index=md['index'], timestamp=md['timestamp'])


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
# Threading, multiprocesing, etc.



def clear_queue(q: queue.Queue) -> None:
    """
    Empty a queue.
    """
    while not q.empty():
        q.get()


def push_queue(q: queue.Queue, elt: Any) -> None:
    """
    Like Queue.put(), but will pop an element prior to put if
    queue is full.
    """
    if q.full():
        q.get()
    q.put(elt)


def read_queue(q: queue.Queue, replace: bool = False) -> List:
    """
    Read a queue's contents by popping its elements until empty.
    If ``replace``  is ``True``, the elements will be pushed back
    onto the queue prior to returning.
    """

    # Pop the elements into a list.
    lst = []
    while not q.empty():
        lst.append(q.get())

    # Optionally put the elements back into the queue.
    if replace:
        for i, elt in enumerate(lst):
            q.put(elt)

    return lst


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
