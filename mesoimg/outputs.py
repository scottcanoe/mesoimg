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
    'FrameBuffer',
    'FramePublisher',
]



class FrameBuffer(io.BytesIO):

    """
    Image buffer for unencoded RGB video.

    """


    def __init__(self,
                 cam: 'Camera',
                 callback: Optional[Callable] = None):
        super().__init__()

        # Basic attributes and their thread lock.
        self._cam = cam
        if callback is None:
            self.callback = cam._write_callback
        else:
            self.callback = callback

        # Initialize reshaping parameters.
        # In raw input mode, the sensor sends us data with resolution
        # rounded up to nearest multiples of 16 or 32. Find this input,
        # which will be used for the initial reshaping of the data.
        fwidth, fheight = raw_resolution(cam.resolution)
        self._in_shape = (fheight, fwidth, 3)

        # Once reshaped, any extraneous rows or columns introduced
        # by the rounding up of the frame shape will need to be
        # sliced off. Additionally, any unwanted channels will
        # need to be removed, so we'll combine the two cropping
        # procedures into one.
        width, height = cam.resolution
        channels = cam.channels
        if channels in ('r', 'g', 'b'):
            ch_index = 'rgb'.find(channels)
            self._out_shape = (height, width)
        else:
            ch_index = slice(None)
            self._out_shape = (height, width, 3)

        if self._in_shape == self._out_shape:
            self._out_slice = (slice(None),   slice(None),  ch_index)
        else:
            self._out_slice = (slice(height), slice(width), ch_index)

        self._n_bytes_in = np.prod(self._in_shape)
        self._n_bytes_out = np.prod(self._out_shape)


    def write(self, data: bytes) -> int:
        """
        Reads and reshapes the buffer into an ndarray, and sets the
        `_frame` attribute with the new array along with its index
        and timestamp.

        Sets the camera's `new_frame` event.
        If dumping to a file and writing is complete, sets
        the camera's `write_complete` event.

        """

        # Write the bytes to the buffer.
        n_bytes = super().write(data)

        # If an entire frame is complete, dispatch it.
        bytes_available = self.tell()
        if bytes_available < self._n_bytes_in:
            print('not full frame', flush=True)
            return n_bytes
        if bytes_available > self._n_bytes_in:
            msg = f"Expected {self._n_bytes_in} bytes, received {bytes_available}"
            raise IOError(msg)

        # Reshape the data from the buffer.
        data = np.frombuffer(self.getvalue(), dtype=np.uint8)
        data = data.reshape(self._in_shape)[self._out_slice]
        data = np.ascontiguousarray(data)
        if self.callback:
            self.callback(data)

        # Finally, rewind the buffer and return as usual.
        self.truncate(0)
        self.seek(0)
        return n_bytes


    def flush(self) -> None:
        super().flush()


    def close(self) -> None:
        self.flush()
        super().close()



class FramePublisher(Thread):

    #complete: Event
    #_closed: bool
    #_path: Path


    def __init__(self,
                 ctx,
                 cam: 'Camera',
                 # cond: Condition,
                 topic: str = 'frame',
                 hwm: int = 10,
                 timeout: float = 1.0,
                 start: bool = True,
                 ):

        super().__init__()

        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(f'tcp://*:{Ports.FRAME_PUB}')
        self.sock.hwm = hwm
        self.topic = topic

        # self.poller = zmq.Poller()
        # self.poller.register(self.sock, zmq.POLLOUT)

        self.cam = cam
        self.timeout = timeout

        if start:
            self.start()


    def run(self):

        self._terminate = False
        cond = self.cam.new_frame
        with cond:
            while not self._terminate:
                ready = cond.wait(self.timeout)
                if not ready:
                    continue
                with self.cam.lock:
                    frame = self.cam.frame
                print(f'Got frame: {frame.index}')
                pub_frame(self.sock, frame)
        self.sock.close()
        time.sleep(0.01)


    def close(self):
        self._terminate = True
        time.sleep(self.timeout + 0.5)













