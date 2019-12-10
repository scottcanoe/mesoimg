import io
from typing import Tuple
import numpy as np


__all__ = [
    'FrameBuffer',
]



def raw_resolution(resolution, splitter=False):
    """
    Round a (width, height) tuple up to the nearest multiple of 32 horizontally
    and 16 vertically (as this is what the Pi's camera module does for
    unencoded output).

    From ``picamera.array``, places here to avoid more picamera imports.

    """

    width, height = resolution
    if splitter:
        fwidth = (width + 15) & ~15
    else:
        fwidth = (width + 31) & ~31
    fheight = (height + 15) & ~15
    return fwidth, fheight



class FrameBuffer(io.BytesIO):

    """
    Image buffer for unencoded RGB video.

    """

    def __init__(self,
                 cam: 'Camera',
                 contiguous: bool = True,
                 ):
        super().__init__()

        self._cam = cam
        self._contiguous = contiguous

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

        # Reshape the data from the buffer, and send the data back
        # to the camera for further handling.
        data = np.frombuffer(self.getvalue(), dtype=np.uint8)
        data = data.reshape(self._in_shape)[self._out_slice]
        if self._contiguous and not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        self._cam._write_callback(data)

        # Finally, rewind the buffer and truncate.
        self.truncate(0)
        self.seek(0)
        return n_bytes


    def flush(self) -> None:
        super().flush()


    def close(self) -> None:
        self.flush()
        self.truncate(0)
        self.seek(0)
        self.data = None
        super().close()



