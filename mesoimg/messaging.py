from enum import IntEnum
import functools
import itertools
import queue
from threading import Condition, Event, Lock, RLock, Thread
import time
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Union
import numpy as np
from superjson import json, SuperJson
import zmq
from mesoimg.arrays import Frame


__all__ = [

    # sockets
    'as_socket_type',
    'create_socket',
    'setsockattr',
    'getsockattr',
    'delsockattr',
    'poll',
    'poll_in',
    'poll_out',
    'can_recv',
    'can_send',

    # send/recv
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

    # Threaded workers
    'RequestReply',
    'Publisher',
]


_SOCK_TYPE_ALIASES = {\
    zmq.PAIR : (zmq.PAIR, 'pair'),
    zmq.REQ  : (zmq.REQ,  'req', 'request'),
    zmq.REP  : (zmq.REP,  'rep', 'reply'),
    zmq.PUSH : (zmq.PUSH, 'push'),
    zmq.PULL : (zmq.PULL, 'pull'),
    zmq.PUB  : (zmq.PUB,  'pub', 'publish'),
    zmq.SUB  : (zmq.SUB,  'sub', 'subscribe'),
}

_TO_SOCK_TYPE = {}
for sock_type, aliases in _SOCK_TYPE_ALIASES.items():
    for val in aliases:
        _TO_SOCK_TYPE[val] = sock_type


def as_socket_type(val: Union[str, int]):
    val = val.lower() if isinstance(val, str) else val
    try:
        return _TO_SOCK_TYPE[val]
    except KeyError:
        msg = f'Invalid argument for socket type: {val}'
        raise TypeError(msg)


def create_socket(sock_type: Union[str, int],
                  poller: Optional[Union[bool, zmq.Poller]] = None,
                  flags: int = zmq.POLLIN | zmq.POLLOUT,
                  timeout: Optional[float] = None,
                  context: Optional[zmq.Context] = None,
                  **kw,
                  ) -> zmq.Socket:


    # Create the socket, and initialize a bunch of attributes to empty values.
    sock_type = as_socket_type(sock_type)
    ctx = context if context else zmq.Context.instance()
    sock = ctx.socket(sock_type)

    # Add polling support.
    if poller in (None, False):
        poller = None
    elif poller is True:
        poller = zmq.Poller()
    elif isinstance(poller, zmq.Poller):
        pass
    else:
        raise ValueError(f'{poller} is not a valid argument for poller.')

    setsockattr(sock, 'poller', poller)
    setsockattr(sock, 'timeout', timeout)
    if poller:
        poller.register(sock, flags=flags)

    # Set a topic filter if given.
    if sock_type == zmq.SUB and 'topic' in kw:
        set_topic(sock, kw['topic'])

    # Set high-water mark if given.
    if 'hwm' in kw:
        sock.hwm = kw['hwm']

    # Finally, return the socket.
    return sock


def set_topic(sock: zmq.Socket, topic: Union[bytes, str]) -> None:
    topic = topic.encode() if isinstance(topic, str) else topic
    sock.setsockopt(zmq.SUBSCRIBE, topic)


def getsockattr(sock: zmq.Socket,
                key: str,
                *default,
                ) -> Any:
    """
    Get a socket's attribute, whether it can be accessed via the socket
    class' __getattr__ or by accessing its attribute dictionary manually.
    """

    # Try attribute access the normal way.
    try:
        return getattr(sock, key)
    except AttributeError:
        pass

    # Try accessing the attribute dict manually.
    try:
        return sock.__dict__[key]
    except KeyError:
        pass

    # Handle default return value.
    N = len(default)
    if N == 0:
        raise AttributeError(f'Socket has no attribute: {key}')
    elif N == 1:
        return default[0]
    else:
        raise TypeError('getsockattr expected 2 or 3 arguments, got {N}')



def setsockattr(sock: zmq.Socket,
                key: str,
                val: Any,
                **kw,
                ) -> None:
    """
    Set a socket's attribute, whether it can be set using the socket
    class' __setattr__ or by modifying its attribute dictionary 'manually'.

    """

    # Handle pollers.
    if key == 'poller':
        cur_poller = getsockattr(sock, 'poller', None)
        if cur_poller:
            cur_poller.unregister(sock)
        if isinstance(val, zmq.Poller):
            val.register(sock, **kw)

    # Try setting the attribute the normal way.
    try:
        return setattr(sock, key, val)
    except AttributeError:
        pass

    # Add to socket's attribute dictionary 'manually'.
    sock.__dict__[key] = val


def delsockattr(sock: zmq.Socket,
                key: str,
                val: Any,
                **kw,
                ) -> None:
    """
    Delete a socket's attribute.
    """

    # Handle pollers.
    if key == 'poller':
        cur_poller = getsockattr(sock, 'poller', None)
        if cur_poller:
            cur_poller.unregister(sock)

    # Try deleting the attribute the normal way.
    try:
        return delattr(sock, key)
    except:
        pass

    # Delete directly from the attribute dict.
    del sock.__dict__[key]


def poll(sock: zmq.Socket,
         flags: int = zmq.POLLIN | zmq.POLLOUT,
         timeout: Optional[float] = None,
         ) -> Optional[Tuple[zmq.Socket, int]]:

    timeout = getsockattr(sock, 'timeout', 0) if timeout is None else timeout
    if timeout is not None:
        timeout *= 1000

    events = dict(sock.poller.poll(timeout))
    if sock in events:
        direction = events[sock]
        if flags == zmq.POLLIN | zmq.POLLOUT or direction == flags:
            return sock, direction
    return None


def poll_in(sock: zmq.Socket,
            timeout: Optional[float] = None,
            ) -> bool:

    res = poll(sock, zmq.POLLIN, timeout=timeout)
    return False if res is None else True


def poll_out(sock: zmq.Socket,
             timeout: Optional[float] = None,
             ) -> bool:

    res = poll(sock, zmq.POLLOUT, timeout=timeout)
    return False if res is None else True


def can_recv(sock: zmq.Socket, poll_result: Dict) -> bool:
    return sock in poll_result and \
           poll_result[sock] in (zmq.POLLIN, zmq.POLLIN | zmq.POLLOUT)


def can_send(sock: zmq.Socket, poll_result: Dict) -> bool:
    return sock in poll_result and \
           poll_result[sock] in (zmq.POLLOUT, zmq.POLLIN | zmq.POLLOUT)



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
    md = {'shape': data.shape,
          'dtype': str(data.dtype),
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



class SocketHandler(Thread):

    """

    """

    _ATTACHMENT_METHOD: ClassVar[str] = ''
    _SOCKET_TYPE: ClassVar[int] = -1


    _socket: zmq.Socket
    _poller: Optional[zmq.Poller] = None
    _start_time: Optional[float] = None
    _stop_time: Optional[float] = None

    _poll_timeout: float = 0.1
    _poll_sleep: float = 0.01
    _linger: Optional[float] = None


    def __init__(self,
                 sock_type: int,
                 send: Optional[Callable] = send_bytes,
                 recv: Optional[Callable] = recv_bytes,
                 context: Optional[zmq.Context] = None,
                 ):

        super().__init__()

        # Synchronization tools.
        self.lock = Lock()
        self._rlock = RLock()
        self._terminate = Event()
        self._run_complete = Event()

        # Set send/recv methods.
        self._send = send
        self._recv = recv

        # Initialize context.
        ctx = context if context else zmq.Context.instance()
        self._socket = ctx.socket(sock_type)


    @property
    def alive(self) -> bool:
        """
        Convenience for Thread.is_alive()
        """
        return self.is_alive()


    def bind(self, addr: str) -> None:
        """
        Bind the socket. Only allowed if thread has not yet started.
        """
        self._check_not_started('Cannot modify socket once thread has started.')
        with self.lock:
            self._socket.bind(addr)


    def connect(self, addr: str) -> None:
        """
        Connect the socket. Only allowed if thread has not yet started.
        """
        self._check_not_started('Cannot modify socket once thread has started.')
        with self.lock:
            self._socket.connect(addr)


    def getsockopt(self, key: str) -> None:
        """
        Get socket options. Only allowed if thread has not yet started.
        """
        self._check_not_started('Cannot access socket once thread has started.')
        with self.lock:
            self._socket.getsockopt(key)


    def setsockopt(self, key: str, val: Any) -> None:
        """
        Set socket options. Only allowed if thread has not yet started.
        """
        self._check_not_started('Cannot modify socket once thread has started.')
        with self.lock:
            self._socket.setsockopt(key, val)


    def getsockattr(self, key: str) -> None:
        """
        Get socket attribute. Only allowed if thread has not yet started.
        """
        self._check_not_started('Cannot access socket once thread has started.')
        with self.lock:
            getsockattr(self._socket, key)


    def setsockattr(self, key: str, val: Any) -> None:
        """
        Set socket attribute. Only allowed if thread has not yet started.
        """
        self._check_not_started('Cannot access socket once thread has started.')
        with self.lock:
            setsockattr(self._socket, key, val)


    def start(self) -> None:
        """
        Record the start time, and start the thread.
        """
        self._start_time = time.time()
        super().start()


    def run(self) -> None:
        """
        Run the event loop.
        - Wait for a request from a socket.
        - On arrival, push to client for action. Then wait for a return
          value to send back to the requester.

        """

        sock = self._socket
        poll_timeout_msecs = self._poll_timeout * 1000
        sleep_secs = 0.001

        while not self._terminate.is_set():
            # poll for input
            if not sock.poll(poll_timeout_msecs):
                time.sleep(sleep_secs)

        # Close sockets,etc.
        self._run_complete.set()
        self._cleanup()
        self._stop_time = time.time()


    def stop(self,
             linger: Optional[float] = None,
             ) -> None:
        """
        Signal the event loop to stop.
        """
        # Request termination of the event loop, and perform timeout check.
        self._terminate.set()


    def _cleanup(self, linger: Optional[float] = None) -> None:
        self._stop_time = time.time()
        self._socket.close(self._linger)



    def _check_not_started(self, msg: str = None):
        if self._started:
            raise RuntimeError(msg)




class RequestReply(SocketHandler):

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

    def __init__(self,
                 sock_type: int,
                 addr: str,
                 requests: Optional[queue.Queue],    # messages received
                 send_q: Optional[queue.Queue] = None,    # messages to send
                 bind: bool = False,
                 connect: bool = False,
                 **kw,
                 ):

        if sock_type not in (zmq.REP, zmq.REQ):
            raise TypeError("sock_type must be zmq.REP or zmq.REQ.")

        super().__init__(sock_type, **kw)

        if bind:
            self._socket.bind(addr)
        elif connect:
            self._socket.connect(addr)

        self._poller = zmq.Poller()
        self._poller.register(self._socket)

        self.inbox = inbox
        self.outbox = outbox


    def run(self) -> None:

        """
        - Wait for a request from a socket.
        - On arrival, push to client for action. Then wait for a return
          value to send back to the requester.

        """

        sock = self._sock
        poller = self._poller
        self.start_time = time.time()

        while not self._terminate.is_set():
            """
            If poll result is empty, then no messages have been sent.

            If result has POLLIN, it means we have a message ready to read.
            """
            poll_result = dict(poller.poll(self._poll_timeout))
            if sock in poll_result and sock[poll_result] == zmq.POLLIN:
                pass

            if can_recv(sock, poll_result):
                # messages are available to read.
                msg = self._recv(sock)
                self.inbox.put(msg)

            if can_send(sock, poll_result):
                try:
                    msg = self.outbox.get(timeout=self.interval)
                except queue.Empty:
                    continue
                self._sender(sock, msg)

            time.sleep(0.005)

        poller.unregister(sock)
        sock.close()
        self._finished.set()
        self.stop_time = time.time()


    def close(self, block: bool = False) -> None:
        self._terminate.set()
        if block:
            self._finished.wait()





class Publisher(Thread):

    """

    Waits for objects in the


    """

    def __init__(self,
                 addr: str,
                 send: Callable,
                 out_q: queue.Queue,
                 topic: str = '',
                 interval: float = 0.005,
                 context: Optional[zmq.Context] = None,
                 ):

        super().__init__()

        # Synchronization tools.
        self.lock = Lock()
        self._terminate = Event()
        self._finished = Event()

        # Initialize socket and poller.
        ctx = context if context else zmq.Context.instance()
        self._sock = ctx.socket(zmq.PUB)
        self._sock.bind(addr)

        # Initialize/set sending and receiving functions and queues.
        self._sender = send
        self.outbox = outbox

        # etc.
        self.interval = interval
        self.start_time = None
        self.stop_time = None


    @property
    def alive(self) -> bool:
        return self.is_alive()


    def run(self) -> None:

        """
        - Wait for a request from a socket.
        - On arrival, push to client for action. Then wait for a return
          value to send back to the requester.

        """

        sock = self._sock
        self.start_time = time.time()

        while not self._terminate.is_set():
            # Wait for an event or whatever.
            try:
                msg = self.outbox.get(timeout=self.interval)
            except queue.Empty:
                continue
            self._sender(sock, msg)

            time.sleep(0.005)

        sock.close()
        self._finished.set()
        self.stop_time = time.time()


    def close(self, block: bool = False) -> None:
        self._terminate.set()
        if block:
            self._finished.wait()









class Publisher(SocketHandler):

    """

    """

    def __init__(self,
                 addr: Optional[str] = None,
                 topic: str = '',
                 bind: bool = True,
                 **kw,
                 ):

        super().__init__(zmq.PUB, **kw)

        self.outbox = outbox

        if bind:
            self._socket.bind(addr)



    def run(self) -> None:

        """
        - Wait for a request from a socket.
        - On arrival, push to client for action. Then wait for a return
          value to send back to the requester.

        """

        sock = self._sock

        while not self._terminate.is_set():
            # Wait for an event or whatever.
            try:
                msg = self.outbox.get(timeout=self._polling_timeout)
            except queue.Empty:
                continue
            self._sender(sock, msg)

            time.sleep(0.005)

        sock.close()
        self._finished.set()
        self.stop_time = time.time()


