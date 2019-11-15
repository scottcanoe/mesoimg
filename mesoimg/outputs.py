from collections import namedtuple
import io
from pathlib import Path
from threading import Lock
from typing import Optional, Union, Sequence
import h5py
import numpy as np
from picamera.array import raw_resolution
from mesoimg.common import Channels, PathLike
from mesoimg.timing import Clock



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
    
    
    
    def __init__(self,
                 cam: 'Camera',
                 out: Optional['FrameStream'] = None,
                 channels: Optional[Union[str, Channels]] = None,                 
                 clock: Optional[Clock] = None,
                 ):
        
        super().__init__()
        
        self._cam = cam
        self._out = out
        self._channels = Channels(channels) if channels else cam.channels
        self._clock = Clock() if clock else cam.clock
        self._frame_counter = 0
        self._frame = None
        self.lock = Lock()
                
        # Initialize reshaping parameters.
        # In raw input mode, the sensor sends us data with resolution
        # rounded up to nearest multiples of 16 or 32. Find this input,
        # which will be used for the initial reshaping of the data.
        fwidth, fheight = raw_resolution(self._cam.resolution)
        self._in_shape = (fheight, fwidth, 3)


        # Once reshaped, any extraneous rows or columns introduced
        # by the rounding up of the frame shape will need to be
        # sliced off. Additionally, any unwanted channels will
        # need to be removed, so we'll combine the two cropping
        # procedures into one.
        width, height = self._cam.resolution
        self._out_shape = (height, width, len(self._channels))

        if self._in_shape == self._out_shape:
            self._out_slice = (slice(None),   slice(None),  self._channels.index)
        else:
            self._out_slice = (slice(height), slice(width), self._channels.index)
        

    
    @property
    def frame(self) -> Frame:
        return self._frame

    
    def write(self, data: bytes) -> int:
        """
        Called by the encoder to write sensor data to the buffer.
        If the frame is complete, sets the `array` attribute
        with the newly read buffer data.
        """
        
        # Write the bytes to the buffer.
        n_bytes = super().write(data)

        # If an entire frame is complete, dispatch it.
        if self._cam.frame.complete:
            
            # Reshape the data from the buffer.
            data = np.frombuffer(self.getvalue(), dtype=np.uint8)
            data = data.reshape(self._in_shape)[self._out_slice]
            frame = Frame(data=data,
                          index=self._frame_counter,
                          timestamp=self._clock())
                        
            # Set attributes, and trigger callback.
            with self.lock:
                self._frame = frame
                self._frame_counter += 1

            # Optionally write to file.
            if self._out:
                self._out.write(frame)
                                    
            # 'Reset' the buffer, and alert camera to new frame.
            self.seek(0)
            self._cam._new_frame_callback(frame)
                
        # Finally, return number of bytes written as usual.
        return n_bytes
        

        
    def flush(self):
        super().flush()
        self._frame = None
        if self._out:
            self._out.flush()
  
    
    def close(self) -> None:
        self.flush()
        super().close()            
        if self._out:
            self._out.close()
        



class FrameStream:

    _closed: bool
    _path: Path
    
    
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

    def close(self):
        raise NotImplementedError



class H5WriteStream(FrameStream):
    

    def __init__(self,
                 path: PathLike,
                 shape: Sequence[int],
                 dtype: Union[str, type] = np.uint8,
                 ):

        self._path = Path(path)        
        self._file = h5py.File(str(self.path), 'w')
        self._closed = False
        
        self._data = self._file.create_dataset('data', shape, dtype=dtype)
        self._data.attrs['n_frames'] = 0
        self._ts = self._file.create_dataset('timestamps', (shape[0],), dtype=float)
        
        self._index = 0
            
            
    @property
    def path(self):
        return self._path
            
    def tell(self):
        return self._data.attrs['index']
    
    def write(self, frame: Frame) -> int:
        """
        The client calls this to dump data.
        """
        
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



