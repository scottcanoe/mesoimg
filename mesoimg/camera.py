from collections import UserDict
import io
import logging
import os
from pathlib import Path
import socket
from threading import Event, Thread
import time
from typing import (Any,
                    Callable,
                    ClassVar,
                    Iterable,
                    List,
                    Mapping,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)

import imageio
import matplotlib.pyplot as plt
import numpy as np
import picamera
from picamera.array import bytes_to_rgb, raw_resolution
from mesoimg.common import *
from mesoimg.timing import Clock, IntervalTimer
from mesoimg.display import ImageViewer
from mesoimg.outputs import H5WriteStream

__all__ = [
    'FrameBuffer',
    'Camera',
]



_CHANNELS = ('r', 'g', 'b', 'rgb')
_CHANNELS_TO_SLICE = {'r'   : 0,
                      'g'   : 1,
                      'b'   : 2,
                      'rgb' : slice(None)}
                                   

class FrameBuffer(io.BytesIO):

    """
    Image buffer for unencoded video.
    
    """
        
    
    def __init__(self, cam: 'Camera'):
        super().__init__()        
        self.cam = cam
        self.array = None            
        self._create_input_handler()

        self.ts = []

    def close(self) -> None:
        super().close()
        self.array = None
        if self.out:
            self.out.flush()
            self.out.close()

    def connect(self, url: URL, maxframes: int) -> None:
        movshape = (maxframes, *self.out_shape)
        self.out = H5WriteStream(url, movshape)
        
        
            

                
    def write(self, data: bytes) -> int:
        """
        Called by the encoder to write sensor data to the buffer.
        If the frame is complete, sets the `array` attribute
        with the newly read buffer data.
        """
        
        # Write the bytes to the buffer.
        n_bytes = super().write(data)        

        # If an entire frame is complete, dispatch it.
        piframe = self.cam.frame
        if piframe.complete:
            arr = np.frombuffer(self.getvalue(), dtype=np.uint8)
            if len(arr) != self.bytes_expected:
                raise RuntimeError('expected different number of bytes')
            self.array = arr.reshape(self.in_shape)[self.out_slice]
            self.seek(0)

            self.ts.append(piframe.timestamp)
            self.out.write(self.array, piframe.timestamp)
            
            #self.send(self.array, info.index, info.timestamp)
                
        # Finally, return number of bytes written as usual.
        return n_bytes
        

    def send(self,
             arr: np.ndarray,
             index: int,
             timestamp: float,
             ) -> None:
        
        # Send to client.
        pass


    def truncate(self, size: Optional[None] = None) -> None:
        if size is not None:
            raise TypeError
        super(FrameBuffer, self).truncate(None)
        raise NotImplementedError


    def flush(self):
        super().flush()
        print('Flush was actually called.')
        self.array = None

    
    def _create_input_handler(self):

        # Check 'channels' spec. can be handled.
        channels = self.cam.channels
        channel_slice = _CHANNELS_TO_SLICE[channels]
                
        # In raw input mode, the sensor sends us data with resolution
        # rounded up to nearest multiples of 16 or 32. Find this input,
        # which will be used for the initial reshaping of the data.
        fwidth, fheight = raw_resolution(self.cam.resolution)
        self.in_shape = (fheight, fwidth, 3)

        # Once reshaped, any extraneous rows or columns introduced
        # by the rounding up of the frame shape will need to be
        # sliced off. Additionally, any unwanted channels will
        # need to be removed, so we'll combine the two cropping
        # procedures into one.
        width, height = self.cam.resolution
        self.out_shape = (height, width, len(channels))

        if self.in_shape == self.out_shape:
            self.out_slice = (slice(None),   slice(None),  channel_slice)
        else:
            self.out_slice = (slice(height), slice(width), channel_slice)

        # etc.
        self.bytes_expected = fwidth * fheight * 3
        
        
    def _create_output_handler(self, out: Any, maxsize = None, **kw) -> None:        
        
        if out is None:
            self.out = None
            self.output_type = None
            self.maxsize = maxsize
            return
        
        # If it's a file, it's either raw or an hdf5. Anything
        # other than an hdf5 will be considered raw.
        if isinstance(out, (str, Path, urllib.parse.ParseResult)):
            
            url = urlparse(out)
            if not url.scheme:
                url = url._replace(scheme='file')
            path = Path(url.path)
            
            # Handle writing to an hdf5 store.        
            if path.suffix().lower() in ('.h5', '.hdf5'):
                dset = url.fragment if url.fragment else 'data'
                dset = dset[1:] if dset.startswith('/') else dset
                if not url.fragment:
                    url._replace(fragment='data')
                
                
                # Open the file, and create the dataset.
                self.file = h5py.File(str(path), 'a')
                if dset in self.file.keys():
                    del self.file[dset]

                # Determine how to initialize the dataset.           
                if maxsize:
                    pass                    
                
                    
            # Handle writing to a raw file.
            else:
            
                pass
                                
                parts = path.split(':')
            
            
        
        
class Camera(picamera.PiCamera):

    """
    Camera that s
    Make a TCP server?

    Parameters
    ----------

    resolution: (int, int)
    framerate: float
    sensor_mode: int
    warm_up float >= 0
        If >0, will used time.sleep() to allow camera to warm up it's sensors..

    """



    _capturing: bool = False
    
    
    def __init__(self,
                 resolution: Tuple[int, int] = (640, 480),
                 framerate: float = 30.0,
                 sensor_mode: int = 7,
                 channels: str = 'rgb',
                 sleep: float = 2.0,
                 **kw):
        
        logging.info('Initializing Camera.')

        super().__init__(resolution=resolution,
                         framerate=framerate,
                         sensor_mode=sensor_mode)
                         
        
        # Set channel settings.
        try:
            self.channels = channels
        except:
            msg = f"Failed to initialize channels with '{channels}'. "
            msg += "Falling back to default '{self._default_channels}'."
            logging.critical(msg)
            self._channels = self._default_channels
        
        if sleep:
            time.sleep(sleep)
            
        
    @property
    def channels(self) -> str:
        return self._channels
            
    @channels.setter
    def channels(self, ch: str) -> None:

        if self._capturing:
            msg  = "Cannot modify 'channels' attribute "
            msg += "while recording/previewing."
            raise RuntimeError(msg)
        
        if ch not in _CHANNELS:            
            msg  = f"'channels' must be one of {_CHANNELS}. "
            msg += f"Not '{ch}'"
            raise ValueError(msg)

        self._channels = ch
        self._n_channels = len(ch)


    @property
    def n_channels(self):
        return self._n_channels
        
                
    @property
    def out_shape(self) -> Tuple:
        """
        Get the frame shape in array dimension order.
        The `channels` dimension will be omitted in single-channel mode.
       
        """        
        n_xpix, n_ypix = self.resolution    
        if self.n_channels > 1:
            return (n_ypix, n_xpix, self.n_channels)
        return (n_ypix, n_xpix)
                
        
        
    def preview(self):
        """
        Stream video to an animated figure, albeit slowly (3-4 frames/sec.).
        Without 
        """
        
        print("Starting preview", flush=True)

        try:

            self._capturing = True
            viewer = ImageViewer(self)
            stream = FrameBuffer(self)
            
            for _ in self.capture_continuous(stream,
                                            'rgb',
                                             use_video_port=True):
                frame = stream.array
                viewer.update(frame)
                if viewer.closed:
                    break

        except Exception as exc:

            if not exc.__class__.__name__ == 'TclError':
                self._capturing = False
                self.close()
                raise
        
        self._capturing = False
        print("Preview finished.", flush=True)

        


    
    def test(self):
        """
        Stream video to an animated figure, albeit slowly (3-4 frames/sec.).
        Without 
        """
        
        self._capturing = True
        stream = ImageBuffer(self)
        #stream = NoBuffer()
        timer = IntervalTimer(verbose=True)
        for foo in self.capture_continuous(stream,
                                           'rgb',
                                           use_video_port=True):
            #frame = stream.array
            timer.tic()                      
            if timer.count > 100:
                break
                                                                                                
        timer.stop()
        self.timer = timer
        self._capturing = False

    


    def __repr__(self):

        if self._camera is None:
            return 'Camera (closed)'

        s  = '       Camera      \n'
        s += '-------------------\n'
        
        attrs = ['resolution',
                 'sensor_mode',
                 'framerate',
                 'exposure_mode',
                 'exposure_speed',
                 'shutter_speed',
                 'awb_mode']
        for key in attrs:
            s += '{}: {}\n'.format(key, getattr(self, key))

        # Report white balance.
        red, blue = [float(val) for val in self.awb_gains]
        s += 'awb_gains: (red={:.2f}, blue={:.2f})\n'.format(red, blue)

        return s





        
    
        
    
        





"""
Need input/output adapter.
--------------------------

- Input will always be a io.BytesIO() object of rgb data.
- Output must always be the green channel only. Shapes may be:
  - 1D when streaming over network.
  - 2D when using local hdf5 storage or preview.
  
- Priorities:
  1) hdf5
    - assess speed
  2) preview
  3) network

- Want server capable of interrupts and remote control.

- Can use PiRGBArray. Confirmed that virtually nothing is lost in doing so.
  There's no way to avoid having to reconfigure the underlying data if we want
  it to be continuous. We can use ndarrays right up until the final pushing of the data
  to its final destination.
  
- If local hdf5 or preview, don't worry about making array contiguous.

- If net streaming, make contiguous, and somehow include metadata.

"""




