"""A thorough test of polling PAIR sockets."""

#-----------------------------------------------------------------------------
#  Copyright (c) 2010 Brian Granger
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file COPYING.BSD, distributed as part of this software.
#-----------------------------------------------------------------------------

import numpy as np
import time
import zmq
from mesoimg.messaging import *


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


def send(data, flags=0, copy=False, track=False):
    pub.send_string(topic)
    md = {'shape' : data.shape, 'dtype' : str(data.dtype)}
    pub.send_json(md, flags | zmq.SNDMORE)
    pub.send(data, flags=flags, copy=copy, track=track)

def recv(sock, flags=0, copy=False, track=False):
    sock.recv()
    md = sock.recv_json(flags=flags)
    data = sock.recv(flags=flags, copy=copy, track=track)
    buf  = memoryview(data)
    arr = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])
    return arr


PORT = 8000
topic = ''
ctx = zmq.Context()

pub = ctx.socket(zmq.PUB)
pub.bind(f'tcp://*:{PORT}')

s1 = ctx.socket(zmq.SUB)
s1.subscribe(topic.encode())
s1.connect(f'tcp://localhost:{PORT}')
s2 = ctx.socket(zmq.SUB)
s2.subscribe(topic.encode())
s2.connect(f'tcp://localhost:{PORT}')

shape = (2, 2)
dtype = np.uint8
arr = np.ones(shape, dtype=dtype)

#a = recv(s1)
#b = recv(s2)
#import queue
#q = queue.Queue()
#s = Publisher(q, send_string)
#s.bind(f'tcp://*:{PORT}')
#s.start()
