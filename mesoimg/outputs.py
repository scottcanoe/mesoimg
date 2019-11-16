from collections import namedtuple
import io
from pathlib import Path
from threading import Event, Lock
from typing import Optional, Union, Sequence
import h5py
import numpy as np
from picamera.array import raw_resolution
from mesoimg.common import PathLike
from mesoimg.timing import Clock, master_clock
import zmq


__all__ = [
    'Frame',
    'FrameBuffer',
    'FrameStream',
    'H5WriteStream',
]




Frame = namedtuple('Frame', ['data',
                             'index',
                             'timestamp'])




class FrameBuffer(io.BytesIO):

    """
    Image buffer for unencoded RGB video.
    
    """
    
    
    def __init__(self, cam: 'Camera'):        
        super().__init__()

        # Basic attributes and their thread lock.
        self._cam = cam    
                        
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
            raise IOError('too many bytes')
                                
        # Reshape the data from the buffer.
        data = np.frombuffer(self.getvalue(), dtype=np.uint8)
        data = data.reshape(self._in_shape)[self._out_slice]

        # Notify camera of new frame.
        self._cam._write_callback(data)
                
        # Finally, rewind the buffer and return as usual.
        self.seek(0)
        return n_bytes
        
        
    def flush(self) -> None:
        super().flush()
      
      
    def close(self) -> None:
        self.flush()
        super().close()
        


class FrameStream:

    #complete: Event
    #_closed: bool
    #_path: Path
        
    def __init__(self, path: PathLike):
        self._path = Path(path)
        self._closed = False
        self.complete = Event()
    
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def path(self) -> Path:
        return self._path
    
    def read(self) -> Frame:
        raise NotImplementedError
    
    def readall(self) -> np.ndarray:
        raise NotImplementedError

    def write(self, frame: Frame) -> int:
        raise NotImplementedError

    def flush(self):
        pass

    def close(self, flush=True):
        raise NotImplementedError



class H5WriteStream(FrameStream):
    

    def __init__(self,
                 path: PathLike,
                 shape: Sequence[int],
                 dtype: Union[str, type],
                 ):

        self._path = Path(path)
        self._file = h5py.File(str(self.path), 'w')
        self._closed = False
                
        self._data = self._file.create_dataset('data', shape, dtype=dtype)
        self._data.attrs['n_frames'] = 0
        self._ts = self._file.create_dataset('timestamps',(shape[0],), dtype=float)
        
        self._index = 0
        self._max_frames = shape[0]
        self.complete = Event()    
        
            
    @property
    def path(self):
        return self._path
            
    
    def tell(self):
        return self._data.attrs['n_frames']
    
    
    def write(self, frame: Frame) -> int:
        """
        The client calls this to dump data.
        """

        # Check for out-of-bounds.
        if self._index >= self._max_frames:
            self.complete.set()
            return 0
        
        # Write the frame and timestamp.
        self._data[self._index] = frame.data
        self._ts[self._index] = frame.timestamp
        
        # Increment counters.
        self._data.attrs['n_frames'] += 1
        self._index += 1
        return 1
    
        
    def flush(self):
        self._file.flush()


    def close(self):
        self.flush()
        self._file.close()
        self._closed = True



