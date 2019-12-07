import io
from pathlib import Path
from threading import Condition, Event, Lock, Thread
import time
from typing import Callable, Optional, Union, Sequence
import h5py
import numpy as np
from picamera.array import raw_resolution
from mesoimg.common import *
import zmq


__all__ = [
    'FramePublisher',
]



class DataSender(Thread):

    """
    - How are we alerted to data that needs sending?
      - by conditions, queues, or events
    - How do we send it?

    The receiver is similar...
    - How are we alerted to data that needs receiving?
       - by polling sockets
    - How do we acquire the data.
      - ?

    """

    def __init__(self,
                 cam: 'Camera',
                 topic: str = 'frame',
                 *,
                 context: Optional[zmq.Context] = None,
                 hwm: int = 10,
                 timeout: float = 1.0,
                 enabled: bool = True,
                 start: bool = True,
                 ):

        super().__init__()

        # Initialize socket.
        ctx = context if context else zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(f'tcp://*:{Ports.FRAME_PUB}')
        self.sock.hwm = hwm
        self.topic = topic

        self.cam = cam
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
        sock = self.sock
        cam = self.cam
        cond = cam.new_frame
        timeout = self.timeout

        self._terminate = False
        cond = self.cam.new_frame
        with cond:
            while not self._terminate:
                if not self._enabled:
                    time.sleep(1)
                    continue
                ready = cond.wait(timeout)
                if not ready:
                    continue
                with cam.lock:
                    frame = cam.frame
                pub_frame(sock, frame)

        sock.close()
        time.sleep(0.01)


    def close(self):
        self._terminate = True




class StatusPublisher(Thread):


    def __init__(self,
                 ctx,
                 cam: 'Camera',
                 topic: str = 'status',
                 hwm: int = 10,
                 timeout: float = 1.0,
                 enabled: bool = True,
                 start: bool = True,
                 ):

        super().__init__()

        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(f'tcp://*:{Ports.STATUS_PUB}')
        self.sock.hwm = hwm
        self.topic = topic

        self.cam = cam
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
        sock = self.sock
        cam = self.cam
        cond = cam.new_frame
        timeout = self.timeout

        self._terminate = False
        cond = self.cam.new_frame
        with cond:
            while not self._terminate:
                if not self._enabled:
                    time.sleep(1)
                    continue
                ready = cond.wait(timeout)
                if not ready:
                    continue
                with cam.lock:
                    status = cam.status
                sock.send_json(status)

        sock.close()
        time.sleep(0.01)


    def close(self):
        self._terminate = True








