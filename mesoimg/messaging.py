import _pickle as pickle
from typing import Any, ClassMethod, Container, Dict, Tuple, Union
import numpy as np
import zmq
from mesoimg.common import Frame


__all__ = [
    'setsockattr',
    'getsockattr',
    'send_array',
    'recv_array',
    'send_bytes',
    'recv_bytes',
    'send_frame',
    'recv_frame',
    'send_json',
    'recv_json',
    'send_pyobj',
    'recv_pyobj',
    'send_string',
    'recv_string',
]



def setsockattr(sock: zmq.Socket,
                key: str,
                val: Any,
                ) -> None:
    sock.__dict__[key] = val


def getsockattr(sock: zmq.Socket,
                key: str,
                *default,
                ) -> Any:
    if default:
        return getattr(sock, key, default[0])
    return getattr(sock, key)


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
    return Frame(data, index=md['index'], timestamp=md['timestamp'])


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

