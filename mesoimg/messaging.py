from enum import IntEnum
import functools
import itertools
import logging
import queue
from threading import Condition, Event, Lock, RLock, Thread
import time
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Union
import numpy as np
from superjson import json, SuperJson
import zmq
from mesoimg.arrays import Frame


logger = logging.getLogger('camera')



__all__ = [

    # send/recv
    'send',
    'recv',
    'send_string',
    'recv_string',
    'send_json',
    'recv_json',
    'send_pyobj',
    'recv_pyobj',
    'send_array',
    'recv_array',
    'send_frame',
    'recv_frame',

    # Threaded workers
    #'RequestReply',
    'SocketThread',
    'Publisher',
    'Subscriber',
    #'DataRelay'
]



def send(socket: zmq.Socket, data: bytes, **kw) -> None:

    """
    Send bytes.
    """
    socket.send(data, **kw)


def recv(socket: zmq.Socket, **kw) -> None:

    """
    Receive bytes.
    """
    return socket.recv(**kw)


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


def send_json(socket: zmq.Socket, data: dict, **kw) -> None:
    """
    Send a dictionary.
    """
    s_data = json.dumps(data)
    socket.send_string(s_data, **kw)


def recv_json(socket: zmq.Socket, **kw) -> Dict:
    """
    Receive a dictionary.
    """
    s_data = socket.recv_string(**kw)
    return json.loads(s_data)


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


def send_array(socket: zmq.Socket, data: np.ndarray, **kw) -> None:

    """
    Send a ndarray.
    """

    md = {'shape' : data.shape, 'dtype' : str(data.dtype)}
    socket.send_json(md, zmq.SNDMORE)
    socket.send(data, **kw)


def recv_array(socket: zmq.Socket, **kw) -> np.ndarray:
    """
    Receive an ndarray over a zmq socket.
    """
    md = socket.recv_json()
    buf  = memoryview(socket.recv(**kw))
    return np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])


def send_frame(socket: zmq.Socket,
               data: Frame,
               flags: int = 0,
               **kw) -> None:
    """
    Send a `Frame` object over a zmq socket.
    """
    md = {'shape': data.shape,
          'dtype': str(data.dtype),
          'index': data.index,
          'timestamp' : data.timestamp}
    socket.send_json(md, flags | zmq.SNDMORE)
    socket.send(data.data, **kw)


def recv_frame(socket: zmq.Socket,
               flags: int = 0,
               **kw) -> Frame:
    """
    Receive a `Frame` object over a zmq socket.
    """

    md = socket.recv_json(flags)
    buf = memoryview(socket.recv(**kw))
    data = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])
    return Frame(data, index=md['index'], timestamp=md['timestamp'])


_SENDERS = {
    bytes : send,
    str : send_string,
    dict : send_json,
    object : send_pyobj,
    np.ndarray : send_array,
    Frame : send_frame,
}

_RECEIVERS = {
    bytes : recv,
    str : recv_string,
    dict : recv_json,
    object : recv_pyobj,
    np.ndarray : recv_array,
    Frame : recv_frame,
}


class SocketThread(Thread):

    """

    """

    socket: zmq.Socket


    def __init__(self,
                 sock_type: int,
                 context: Optional[zmq.Context] = None,
                 ):
        super().__init__()

        # Threading tools
        self.lock = Lock()
        self.rlock = RLock()
        self.terminate = Event()

        # Create socket.
        ctx = context if context else zmq.Context.instance()
        self._socket = ctx.socket(sock_type)



    @property
    def alive(self) -> bool:
        """
        Convenience for Thread.is_alive()
        """
        return self.is_alive()

    @property
    def closed(self) -> bool:
        """
        See whether the socket is closed.
        """
        return self._socket.closed

    @property
    def rcvtimeo(self) -> int:
        """
        """
        return self._socket.rcvtimeo


    @rcvtimeo.setter
    def rcvtimeo(self, msecs: int) -> None:
        with self.lock:
            self._socket.rcvtimeo = int(msecs)


    def bind(self, addr: str) -> None:
        """
        Bind the socket. Only allowed if thread has not yet started.
        """
        with self.lock:
            self._socket.bind(addr)


    def connect(self, addr: str) -> None:
        """
        Connect the socket. Only allowed if thread has not yet started.
        """
        with self.lock:
            self._socket.connect(addr)


    def getsockopt(self, key: str) -> None:
        """
        Get socket options. Only allowed if thread has not yet started.
        """
        self._socket.getsockopt(key)


    def setsockopt(self, key: str, val: Any) -> None:
        """
        Set socket options. Only allowed if thread has not yet started.
        """
        with self.lock:
            self._socket.setsockopt(key, val)



    def run(self) -> None:
        """
        Run the event loop.
        - Wait for a request from a socket.
        - On arrival, push to client for action. Then wait for a return
          value to send back to the requester.

        """
        try:
            while not self.terminate.is_set():
                time.sleep(0.1)
        except:
            self._cleanup(linger=0)
            raise

        self._cleanup()


    def stop(self) -> None:
        """
        Signal the event loop to stop with the terminate event.
        """
        # Request termination of the event loop, and perform timeout check.
        self.terminate.set()


    def close(self) -> None:
        """
        Same as ``stop``.
        """
        self.terminate.set()


    def _cleanup(self, linger: Optional[int] = None) -> None:
        """
        Other threads shouldn't call this. The call to this should
        be at the end of `run()`.
        """
        with self.rlock:
            self._socket.close(linger)



class Publisher(SocketThread):
    """
    Publishing socket/thread.


    """

    def __init__(self,
                 send: Callable = send,
                 source: Optional[queue.Queue] = None,
                 topic: Union[bytes, str] = b'',
                 context: Optional[zmq.Context] = None,
                 **send_kw,
                 ):

        super().__init__(zmq.PUB, context)

        self._send = send
        self._send_kw = send_kw
        self._source = source
        self.topic = topic


    @property
    def source(self) -> queue.Queue:
        return self._source

    @source.setter
    def source(self, obj: Optional[queue.Queue]) -> None:
        with self.lock:
            self._source = obj

    @property
    def topic(self) -> bytes:
        return self._topic

    @topic.setter
    def topic(self, topic) -> None:
        with self.lock:
            topic = topic.encode() if isinstance(topic, str) else topic
            self._topic = topic

    def emit(self,
             item: Any,
             block: bool = True,
             timeout: Optional[float] = None,
             ) -> None:

        if self._source is None:
            raise RuntimeError('Emitter has no source queue.')
        self._source.put(item, block=block, timeout=timeout)


    def run(self) -> None:

        """
        - Wait for a request from a socket.
        - On arrival, push to client for action. Then wait for a return
          value to send back to the requester.

        """

        if self._source is None:
            with self.lock:
                self._source = queue.Queue()

        try:

            while not self.terminate.is_set():

                # Get something...
                try:
                    data = self._source.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Send it...
                with self.lock:
                    self._socket.send(self._topic, zmq.SNDMORE)
                    self._send(self._socket, data, **self._send_kw)

        except:
            self._cleanup(0)
            raise

        self._cleanup()



class Subscriber(SocketThread):
    """


    """

    def __init__(self,
                 recv: Callable = recv,
                 callback: Optional[Callable] = None,
                 context: Optional[zmq.Context] = None,
                 **recv_kw,
                 ):

        super().__init__(zmq.SUB, context)

        self._recv = recv
        self._recv_kw = recv_kw
        self._socket.subscribe(b'')
        self._socket.rcvtimeo = 1000
        self._callback = callback


    @property
    def callback(self) -> Callable:
        return self._callback

    @callback.setter
    def callback(self, fn: Callable) -> None:
        with self.lock:
            self._callback = fn


    def subscribe(self, topic: Union[bytes, str]) -> None:
        topic = topic.encode if isinstance(topic, str) else topic
        with self.lock:
            self._socket.subscribe(topic)


    def unsubscribe(self, topic: Union[bytes, str]) -> None:
        topic = topic.encode if isinstance(topic, str) else topic
        with self.lock:
            self._socket.unsubscribe(topic)


    def run(self) -> None:

        """
        - Wait for a request from a socket.
        - On arrival, push to client for action. Then wait for a return
          value to send back to the requester.

        """
        try:

            while not self.terminate.is_set():

                # Get something...
                try:
                    topic = self._socket.recv()
                    data = self._recv(self._socket, **self._recv_kw)
                except zmq.error.Again:
                    continue

                # Do something with it...
                with self.lock:
                    if self._callback:
                        self._callback(data)

        except:
            self._cleanup()
            raise

        self._cleanup()




    """



                    state |  REQ  |  REP    Waiting for:
                   -------+-------+-------+
                   |  0   |  OUT  |   -   |  new msg  /  partner
      REQ sends -> |      |       |       |
                   |  1   |   -   |   IN  |  partner  /  recv
   REP receives -> |      |       |       |
                   |  2   |   -   |  OUT  |  partner  /  new msg
     REP sends ->  |      |       |       |
                   |  3   |  IN   |       |  recv     /  partner
   REQ receives -> |      |       |       |
                   | (0)  |  OUT  |       |

                      .
                      .
                      .

    When you receive, you switch from POLLIN to POLLOUT.
    When you send a message, you are neither POLLIN or POLLOUT (until its read)

    REP/REQ sockets should never be both POLLIN and POLLOUT. Pair sockets
    can do that though.

"""

#class Reply(SocketThread):


    #def __init__(self,
                 #recv,     # receive input,
                 #send,
                 #inbox,    # push it onto the out_q
                 #outbox,     # get the response from the in_q,
                 #timeout=0.01,
                 #**kw,
                 #):

        #super().__init__(zmq.REQ, **kw)
        #self.poller = zmq.Poller()
        #self.poller.register(self._socket)


    #def run(self) -> None:

        #"""
        #- Wait for a request from a socket.
        #- On arrival, push to client for action. Then wait for a return
          #value to send back to the requester.

        #"""

        #sock = self._socket
        #poller = self.poller
        #timeout = self.timeout

        #while not self.terminate.is_set():
            #"""
            #If poll result is empty, then no messages have been sent.

            #If result has POLLIN, it means we have a message ready to read.
            #"""
            #info = dict(poller.poll(timeout))
            #res = info.get(sock)
            #if res is None:
                #intmo =

            #if info.get(sock) == zmq.POLLIN:
                ## Get from

            #if sock in poll_result and sock[poll_result] == zmq.POLLIN:
                #pass

            #if can_recv(sock, poll_result):
                ## messages are available to read.
                #msg = self._recv(sock)
                #self.inbox.put(msg)

            #if can_send(sock, poll_result):
                #try:
                    #msg = self.outbox.get(timeout=self.interval)
                #except queue.Empty:
                    #continue
                #self._sender(sock, msg)

            #time.sleep(0.005)

        #poller.unregister(sock)
        #sock.close()
        #self._finished.set()
        #self.stop_time = time.time()


    #def close(self, block: bool = False) -> None:
        #self._terminate.set()
        #if block:
            #self._finished.wait()



