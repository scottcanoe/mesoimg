import queue
from queue import Queue
from threading import Lock, Thread
import time
from typing import Callable, Optional, Union, Sequence, NamedTuple
import numpy as np
import zmq
from mesoimg.common import *


HOST = 'pi-meso.local'


class Subscriber(Thread):

    #: Wrapped zmq socket.
    socket: zmq.Socket

    #: Socket's poller for non-blocking I/O.
    poller: zmq.Poller

    #: The queue onto which newly arrived data will be pushed onto.
    q: Queue

    #: How long to wait for new data for each iteration of the event loop.
    _timeout: float = 1.0

    #: Whether we're running the event loop.
    _running: bool = False

    #: Whether to stop the thread at soonest convenience.
    _terminate: bool = False

    def __init__(self,
                 ctx: Optional[zmq.Context],
                 host: str,
                 port: Union[int, str],
                 topic: Union[bytes, str],
                 recv: Callable,
                 q: Queue,
                 hwm: int = 1000,
                 timeout: float = 1.0,
                 start: bool = False,
                 ):

        super().__init__()

        if ctx is None:
            ctx = zmq.Context.instance()

        self.sock = ctx.socket(zmq.SUB)
        topic = topic.encode() if isinstance(topic, str) else topic
        self.sock.subscribe = topic
        self.sock.hwm = hwm
        self.sock.connect(f'tcp://{host}:{port}')

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self.timeout = timeout


        self.q = q
        self.lock = Lock()

        self._terminate = False
        if start:
            self.start()


    def run(self):

        # Alias
        sock = self.sock
        poller = self.poller
        timeout = self.timeout

        self._terminate = False
        while not self._terminate:

            ready = dict(poller.poll(timeout * 1000))
            if sock in ready and ready[sock] == zmq.POLLIN:
                frame = sub_frame(sock)
                if self.q.full():
                    self.q.get(timeout=0.005)
                self.q.put(frame)
                time.sleep(0.005)
                if self.verbose:
                    print(f'Received frame: {frame.index}', flush=True)

        sock.close()
        poller.unregister(sock)
        time.sleep(0.01)


    def close(self):
        self._terminate = True



class FrameSubscriber(Thread):

    verbose = False

    def __init__(self,
                 ctx,
                 q: Queue,
                 topic: str = 'frame',
                 hwm: int = 10,
                 timeout: float = 1.0,
                 enabled: bool = True,
                 start: bool = True,
                 ):

        super().__init__()


        self.sock = ctx.socket(zmq.SUB)
        self.sock.subscribe = topic.encode()
        self.sock.hwm = hwm
        self.sock.connect(f'tcp://{HOST}:{Ports.FRAME_PUB}')
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        self.timeout = timeout

        self.q = q

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
        sock = self.sock
        poller = self.poller
        timeout = self.timeout

        self._terminate = False
        while not self._terminate:
            if not self._enabled:
                time.sleep(1)
            ready = dict(poller.poll(timeout * 1000))
            if sock in ready and ready[sock] == zmq.POLLIN:
                frame = sub_frame(sock)
                if self.q.full():
                    self.q.get(timeout=0.005)
                self.q.put(frame)
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

