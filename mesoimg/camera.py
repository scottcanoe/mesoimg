from collections import UserDict
import io
import logging
import os
from pathlib import Path
import socket
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
from picamera.array import PiRGBArray
from mesoimg.common import (ArrayTransform,
                            PathLike,
                            uint8,
                            pathlike,
                            read_json,
                            write_json,
                            squeeze)
from mesoimg.timing import Clock, Timer
from mesoimg.display import ImageViewer









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


    default_channels: ClassVar[str] = 'rgb'

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
                         
        self._capturing = False
        
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
        
        if ch not in ('r', 'g', 'b', 'rgb'):
            msg = "'channels' must be one of 'r', g', 'b', "
            msg += f"'rgb'. Not '{ch}.'"
            raise ValueError(msg)

        self._channels = ch
        
                
    @property
    def frame_shape(self) -> Tuple:
        """
        Get the frame shape in array dimension order.
        The `channels` dimension will be omitted in single-channel mode.
       
        """        
        n_xpix, n_ypix = self.resolution        
        if len(self._channels) > 1:
            return (n_ypix, n_xpix, len(self._channels))
        return (n_ypix, n_xpix)
                

    def preview(self):
        
        print("Starting preview", flush=True)

        try:

            self._capturing = True
            shape = self.frame_shape
            stream = io.BytesIO()
            viewer = ImageViewer(self)

            for foo in self.capture_continuous(stream,
                                              'rgb',
                                               use_video_port=True):

                frame = np.frombuffer(stream.getvalue(), dtype='u1')
                frame = frame.reshape(shape)
                viewer.update(frame)
                stream.seek(0)
                
                if viewer.closed:
                    break
                                                                                
        except Exception as exc:

            if not exc.__class__.__name__ == 'TclError':
                self._capturing = False
                stream.seek(0)
                self.close()
                raise
        
        self._capturing = False
        stream.seek(0)
        print("Preview finished.", flush=True)
        
                        
    
    
    def capture(self,
                out: Union[None, PathLike] = None,
                show: bool = False,
                **kw):


        config = self.config.copy()
        if kw:
            config.update(kw)
        
        out_shape = self._get_out_shape(out, config)
        out_size = np.prod(out_shape)

        # - No output specified, save to buffer (not to disk)
        if out is None:
            out = PiRGBArray(self)
            #out = PiRGBArray(self, size=out_size)
            format = 'rgb'
            
        # - A path was specified, figure out the format.
        elif pathlike(out):
            out = Path(out)
            ext = out.suffix.lower()
            if ext in ('.h5', '.hdf5'):
                format = 'rgb'
                raise NotImplementedError
            
            format = None
            out = str(out)
            
        # - Something else (a network socket?) was specified...
        else:
            raise NotImplementedError

        picamera.PiCamera.capture(self,
                                  out,
                                  format=format,
                                  use_video_port=config.use_video_port)

        if show:
            ImageViewer(out)

        return out
        
        
    def record(self,
               out: Union[PathLike, socket.socket],     # Filename or socket.
               tmax: float,
               signals: Optional[List[str]] = None,
               **kw):
            
        
        # Get basic config from args.
        config = self.config.copy()
        config.update(kw)
        
        # Estimate the number of frames that will be captured.
        max_frames = int(np.ceil(tmax * self.framerate))
        
        
        # Setup buffer and other configs.
        buf = picamera.PiRGBArray()
        format = 'rgb'
        use_video_port = config['use_video_port']

        if config.raw:
            n_channels = len(config.channels)
            
        # Setup output handler.
        
        #- Write to local file. 
        if pathlike(out):
            if Path(out).suffix in ('.h5', '.hdf5'):
                raise NotImplementedError
            else:
                buf = out
                output_handler = None

        #- Stream over network.
        elif isinstance(out, tuple) and len(out) == 2:
            host, port = out
            
            raise NotImplementedError
        
        # Setup state variables, and begin.
        frame_counter = 0
        timestamps = np.zeros(max_frames)        
        signals = [] if signals is None else signals
        df = {sig: np.zeros(max_frames) for sig in signals}
        
        logging.info('Beginning recording.')
        try:
            clock, Timer = Clock(), Timer()
            tm.start()
            for foo in self.capture_continuous(buf,
                                               format=format,
                                               use_video_port=use_video_port):
                
                # Check for interrupts.
                #self.poll()
                
                frame_counter += 1
                t = clock.time()
    
                # Check frames are in-bounds.
                if frame_counter > max_frames:
                    raise RuntimeError('Exceeded max. frame estimate. Quitting.')
                
                # Check time is in-bounds.
                if t > tmax:
                    break
    
                # Grab the buffered data, and remove non-green channels.
                arr = stream.getvalue()
                stream.seek(0)
    
                # Keep only green chanel.
                arr = arr[1::3]
    
                # Update frames and timestamps.
                frames.append(arr)
                timestamps.append(ts)
                analog_gains.append(cam.analog_gain)
                awb_gains.append(cam.awb_gains)
                exposure_speeds.append(cam.exposure_speed)
                
                if frame_counter == 30:
                    cam.exposure_mode = 'off'
                    cam.analog_gain = 5
                    cam.awb_mode = 'off'
    
                if frame_counter > 3:
                    cam.awb_gains = (1.25, 1.65)
                
        except:
            cam.close()
            clock.stop()
            raise
    
        if cam_created:
            cam.close()
        clock.stop()
    
        if verbose:
            MesoCam.print_timing_summary(timestamps)
    
        timestamps = np.array(timestamps)
        frames = [np.frombuffer(fm, dtype=uint8) for fm in frames]
        
        d = {
             'frames'    : frames,
             'timestamps': timestamps,
             'analog_gains' : analog_gains,
             'awb_gains' : awb_gains,
             'exposure_speeds' : exposure_speeds,
             }
        
        return d        
             
                    

        

    #def _get_format(self, out: Any, config: Config) -> str:
    #    pass

        
    def _get_out_shape(self, out: Any) -> Sequence[int]:

        n_xpix, n_ypix = self.resolution
        if not config.raw or config.channels == 'rgb':
            return [n_ypix, n_xpix, 3]
        else:
            return [n_ypix, n_xpix]

    
    
    def _get_channel_extract(self) -> Optional[ArrayTransform]:
        
        if not config.raw or config.channels == 'rgb':
            return None

        channels = config.channels
        assert channels in 'rgb'
        ix = 'rgb'.find(channels)
        return lambda arr : arr[:, :, ix]
            
 
            

    @staticmethod
    def print_timing_summary(timestamps):

        n_frames = len(timestamps)

        if n_frames == 0:
            print('No timestamps recorded.')
            return
        elif n_frames == 1:
            print('Only one timestamps recorded ({}).'.format(timestamps[0]))
            return

        T = timestamps[-1]
        IFIs = np.ediff1d(timestamps)
        print('n_frames: {}'.format(n_frames))
        print('secs: {:.2f}'.format(T))
        print('FPS: {:.2f}'.format(n_frames / T))
        print('median IFI: {:.2f} msec.'.format(1000 * np.median(IFIs)))
        print('max IFI: {:.2f} msec.'.format(1000 * np.max(IFIs)))
        print('min IFI: {:.2f} msec.'.format(1000 * np.min(IFIs)))



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




