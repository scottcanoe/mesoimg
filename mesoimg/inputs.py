import queue
from queue import Queue
from threading import Lock, Thread
import time
from typing import Callable, Optional, Union, Sequence, NamedTuple
import numpy as np
import zmq
from mesoimg.common import *


HOST = 'pi-meso.local'


class DataReceiver(Thread):

    """
    Must specify:
      - How to be alerted of data to be sent (wait for a condition or queue?)
      - How to send it.



    """
    #: Wrapped zmq socket.
    _socket: zmq.Socket

    #: Socket's poller for non-blocking I/O.
    _poller: zmq.Poller

    #: The queue onto which newly arrived data will be pushed onto.
    _q: queue.Queue

    #: How long to wait for new data for each iteration of the event loop.
    _timeout: float

    #: Internal flag indicating whether networking interface is open.
    _closed: bool

    #: Internal flag checked on iterations of the event loop.
    #: Subscriber.close() sets this to ``True`` to schedule closing.
    _terminate: bool


    def __init__(self,
                 host: str,
                 port: Union[int, str],
                 topic: Union[bytes, str],
                 handler: Optional[Callable] = None,
                 *,
                 context: Optional[zmq.Context] = None,
                 hwm: int = 1000,
                 timeout: float = 1.0,
                 start: bool = False,
                 ):

        super().__init__()

        # Setup socket.
        ctx = context if context else zmq.Context()
        self._sock = ctx.socket(zmq.SUB)
        topic = topic.encode() if isinstance(topic, str) else topic
        self._sock.subscribe = topic
        self._sock.hwm = hwm
        self._sock.connect(f'tcp://{host}:{port}')

        # Setup poller.
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        self._timeout = timeout

        # Initialize internal flags.
        self._closed = False
        self._terminate = False

        # Optionally start the thread.
        if start:
            self.start()


    def run(self):

        # Alias
        sock = self._sock
        poller = self._poller
        timeout = self._timeout * 1000
        msec = 0.001

        while not self._terminate:
            ready = dict(poller.poll(timeout))
            if sock in ready and ready[sock] == zmq.POLLIN:
                self.handler()
                self._handle_recv()
                frame = sub_frame(sock)
                qput(frame)
            time.sleep(msec)

        sock.close()
        poller.unregister(sock)
        self._closed = True


    def close(self,
              block: bool = True,
              pause: float = 0.005,
              ) -> None:

        self._terminate = True
        if block:
            while not self._closed:
                time.sleep(pause)


    def _handle_recv(self):
        raise NotImplementedError



class FrameSubscriber(Thread):

    verbose = False

    def __init__(self,
                 q: Queue,
                 topic: str = 'frame',
                 *,
                 context: Optional[zmq.Context] = None,
                 start: bool = True,
                 hwm: int = 10,
                 timeout: float = 1.0,
                 ):

        super().__init__()

        ctx = context if context else zmq.Context.instance()

        self._sock = ctx.socket(zmq.SUB)
        self._sock.subscribe = topic.encode()
        self._sock.hwm = hwm
        self._sock.connect(f'tcp://{HOST}:{Ports.FRAME_PUB}')

        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        self._timeout = timeout

        self._q = q

        self._terminate = False
        if start:
            self.start()


    def run(self):

        # Alias
        sock = self._sock
        poller = self._poller
        q = self._q
        timeout_msecs = self._timeout * 1000
        msec = 0.001

        self._terminate = False
        while not self._terminate:
            ready = dict(poller.poll(timeout_msecs))
            if sock in ready and ready[sock] == zmq.POLLIN:
                frame = sub_frame(sock)
                if q.full():
                    q.get()
                q.put(frame)
                time.sleep(0.005)
                if self.verbose:
                    print(f'Received frame: {frame.index}', flush=True)

        sock.close()
        poller.unregister(sock)
        time.sleep(0.01)


    def close(self):
        self._terminate = True





class FrameConsumer(Thread):

    # belongs to client, and is modified by frame subscriber.
    q: Queue

    verbose = False

    def __init__(self,
                 q: Queue,
                 timeout: float = 1.0,
                 enabled: bool = True,
                 start: bool = True,
                 ):

        super().__init__()

        self.q = q
        self.timeout = timeout
        self.enabled = enabled
        self._terminate = False
        if start:
            self.start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, tf: bool) -> None:
        assert isinstance(tf, bool)
        self._enabled = True


    def run(self):

        # Alias
        timeout = self.timeout

        self._terminate = False
        while not self._terminate:
            if not self._enabled:
                time.sleep(1)
            try:
                frame = self.q.get(timeout=timeout)
            except queue.Empty:
                continue

            # Do... something with the frame.


            if self.verbose:
                print(f'Received frame: {frame.index}', flush=True)



    def close(self):
        self._terminate = True



class H5Receiver(Thread):

    """
    Receives frames from frame socket. Only stores 1-most recent.

    """

    def __init__(self, frame_sock, path, shape, dtype=np.uint8):
        super().__init__()
        self.frame_sock = frame_sock
        self.lock = Lock()
        self.frame = None
        self.n_received = 0
        self.t_last = None
        self.terminate = False

        self.path = Path(path)
        self.file = h5py.File(str(self.path), 'w')

        self.max_frames = shape[0]
        self.dset = self.file.create_dataset('data', shape, dtype=dtype)
        self.dset.attrs['index'] = 0
        self.ts = self.file.create_dataset('timestamps', (shape[0],), dtype=float)

        self.complete = False
        self.terminate = False





    def run(self):

        global frame
        global n_received

        frame = None
        n_received = 0

        while not self.terminate:

            fm = recv_frame(self.frame_sock)
            frame = fm
            n_received += 1

            with self.lock:
                if not self.complete:
                    index = self.dset.attrs['index']
                    self.dset[index, ...] = fm.data
                    self.ts[index] = fm.timestamp
                    self.dset.attrs['index'] += 1
                    if self.dset.attrs['index'] >= self.max_frames:
                        self.complete = True
                        self.file.close()
                        self.terminate = True
                    self.frame = fm
                    self.n_received += 1

