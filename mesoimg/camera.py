from collections import UserString
import io
import logging
import os
from pathlib import Path
import socket
from threading import Event, Lock, Thread
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
import numpy as np
import picamera
from mesoimg.common import *
from mesoimg.timing import Clock, IntervalTimer
from mesoimg.outputs import Frame, FrameBuffer, FrameStream, H5WriteStream
from mesoimg.errors import *


__all__ = [
    'Camera',
]



        
        
class Camera:

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

    _config_vars = [
        'resolution',
        'channels',
        'framerate',
        'sensor_mode',
        'exposure_mode',
        'exposure_speed',
        'analog_gain',
        'digital_gain',
        'awb_mode',
        'awb_gains',
        'iso',
        'shutter_speed',        
    ]
    
    config = None
    
    def __init__(self,
                 resolution: Tuple[int, int] = (640, 480),
                 channels: str = 'g',
                 framerate: float = 30.0,
                 sensor_mode: int = 7,
                 sleep: float = 2.0,
                 **kw):
                        
        # Events, flags, locks.
        self.lock = Lock()
        self.frame = None   
        self.recording = False
        self.write_complete = False
        self.abort = False
                                        
        # Attributes.
        self.channels = channels
                        
        # The picamera instance.
        logging.info('Initializing Camera.')
        self._cam = picamera.PiCamera(resolution=resolution,
                                      framerate=framerate,
                                      sensor_mode=sensor_mode)
        if sleep:
            time.sleep(sleep)
            
        self.stash_config()
        
        #self.awb_mode = awb_mode
        
        self._closed = False
        
        
        
    #-----------------------------------------------------------#
    # Main camera settings

    @property
    def resolution(self) -> Tuple[int, int]:
        res = self._cam.resolution
        return (int(res[0]), int(res[1]))

    @resolution.setter
    def resolution(self, res: Tuple[int, int]) -> None:
        self._cam.resolution = res

    @property
    def framerate(self) -> float:
        return float(self._cam.framerate)

    @framerate.setter
    def framerate(self, fps: float) -> None:
        self._cam.framerate = fps

    @property
    def sensor_mode(self) -> int:
        return self._cam.sensor_mode
        
    @sensor_mode.setter
    def sensor_mode(self, mode: int) -> None:
        self._cam.sensor_mode = mode
           
    @property
    def awb_mode(self) -> str:
        return self._cam.awb_mode
        
    @awb_mode.setter
    def awb_mode(self, mode: str) -> None:
        self._cam.awb_mode = mode
                
    @property
    def awb_gains(self) -> Tuple[float, float]:
        gains = self._cam.awb_gains
        return (float(gains[0]), float(gains[1]))
        
    @awb_gains.setter
    def awb_gains(self, gains: Tuple[float, float]) -> None:
        self._cam.awb_gains = gains
    
    @property
    def iso(self) -> int:
        return self._cam.iso
        
    @iso.setter
    def iso(self, val: int) -> None:
        self._cam.iso = val

    @property
    def exposure_mode(self) -> str:
        return self._cam.exposure_mode
        
    @exposure_mode.setter
    def exposure_mode(self, mode: str) -> None:
        self._cam.exposure_mode = mode

    @property
    def analog_gain(self) -> float:
        return float(self._cam.analog_gain)
        
    @property
    def digital_gain(self) -> float:
        return float(self._cam.digital_gain)

    @property
    def exposure_speed(self) -> int:
        return self._cam.exposure_speed
        
    @exposure_speed.setter
    def exposure_speed(self, speed: int) -> None:
        self._cam.exposure_speed = speed

    @property
    def shutter_speed(self) -> int:
        return self._cam.shutter_speed
        
    @shutter_speed.setter
    def shutter_speed(self, speed: int) -> None:
        self._cam.shutter_speed = speed

    @property
    def closed(self):
        return True if self._cam is None else self._cam.closed

        
    #-----------------------------------------------------------#
    # My settings, and some convenience properties.
    

    @property
    def channels(self) -> str:
        return self._channels
            
    @channels.setter
    def channels(self, ch: str) -> None:
        if ch not in ('r', 'g', 'b', 'rgb'):
            raise ValueError(f"invalid channels '{ch}'")
        self._channels = ch
    
    
    @property
    def out_shape(self) -> Tuple:
        """
        Get the frame shape in array dimension order.
        The `channels` dimension will be omitted in single-channel mode.
       
        """        
        n_xpix, n_ypix = self.resolution
        n_channels = len(self.channels)
        if n_channels > 1:
            return (n_ypix, n_xpix, n_channels)
        return (n_ypix, n_xpix)
                
    
    #-----------------------------------------------------------#
    # Public methods.

    def record(self,
               outfile: PathLike,
               duration: float,
               overwrite: bool = False,
               ) -> FrameBuffer:
        
        """
        Record for a fixed period of time.
        """
        
        if self.recording or self.write_complete or self.abort:
            raise RuntimeError
        
        # Setup outfile and frame buffer.
        outfile = self._init_outfile(outfile, duration, overwrite)
        out = FrameBuffer(self, outfile=outfile)
        
        self.start_recording(out)
        while True:
            if self.write_complete or self.abort:
                break
            self.wait_recording(1.0)
        self.stop_recording()
        
        # Flush and close files/buffers.
        with self.lock:
            out.close()
            outfile.close()
            self.write_complete = False
            self.abort = False

        
                
    def start_recording(self,
                        out: Optional[FrameBuffer] = None,
                        ) -> FrameBuffer:
        """
        Start recording given a frame buffer, or optionally
        one will be created for you.
        """

        out = out if out else FrameBuffer(self)
        with self.lock:
            self.recording = True
        print('Starting recording', flush=True)
        self._cam.start_recording(out, 'rgb')
        return out
        
       
    def wait_recording(self, timeout: float = 0) -> None:
        self._cam.wait_recording(timeout)
        
        
    def stop_recording(self) -> None:

        self._cam.stop_recording()
        with self.lock:
            self.recording = False
        print('Recording stopped.', flush=True)   


    def preview(self):
        """
        Stream video to an animated figure, albeit slowly (3-4 frames/sec.).
        Without 
        """
        
        from mesoimg.display import Preview
        
        print("Starting preview", flush=True)

        try:

            self.frame_buffer = FrameBuffer(self)
            self.running.set()
            viewer = Preview(self, self.frame_buffer)
            self.start_recording(self.frame_buffer, 'rgb')
            while not viewer.closed:
                self.wait_recording(1)
            self.stop_recording()

        except Exception as exc:

            if not exc.__class__.__name__ == 'TclError':
                self.cleanup()
                self.close()
                raise

        self.cleanup()
        print("Preview finished.", flush=True)


    def cleanup(self) -> None:
        with self.lock:
            self.recording = False
            self.write_complete = False
            self.abort = False
            
    def close(self) -> None:
        self.cleanup()        
        if self._cam:
            if not self._cam.closed:
                self.stash_config()
            self._cam.close()
            self._cam = None

    def stash_config(self) -> None:
        config = {}
        for name in self._config_vars:
            config[name] = getattr(self, name)
        self.config = config
    
    def reopen(self) -> None:
        pass
        
                                        
    #-----------------------------------------------------------#
    # Private/protected methods.
    

    def _init_outfile(self,
                      path: PathLike,
                      duration: float,
                      overwrite: bool = False,
                      ) -> None:

        path = Path(path)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(path)

        ext = path.suffix.lower()
        if ext not in ('.h5', '.hdf5'):
            raise NotImplementedError(f'invalid file path: {str(path)}')
                
        n_frames = int(np.ceil(self.framerate * duration))        
        shape = (n_frames, *self.out_shape)
        outfile = H5WriteStream(path, shape, dtype=np.uint8)
        return outfile

    
    def _write_callback(self, frame: Frame) -> None:

        with self.lock:
            self.frame = frame
        
        if read_stdin().strip() == 'q':
            print('user abort', flush=True)
            with self.lock:
                self.abort = True
                        

    def __repr__(self):

        if self.closed:
            return 'Camera (closed)'

        s  = '       Camera      \n'
        s += '-------------------\n'
        s += f'sensor mode: {self.sensor_mode}\n'
        s += f'resolution: {self.resolution}\n'
        s += f'channels: {str(self.channels)}\n'
        s += f'framerate (?): {self.framerate}\n'
        
        s += f'exposure mode: {self.exposure_mode}'
        if self.exposure_mode == 'off':
            line += ' (gain locked at current values)'
        s += '\n'
        
        s += f'analog gain: {float(self.analog_gain)}\n'
        s += f'digital gain: {float(self.digital_gain)}\n'
        s += f'ISO : {self.iso}\n'
        s += f'exposure speed: {self.exposure_speed} microsec.\n'

        

        # Report auto-white balance.
        s += f'awb mode: {self.awb_mode}\n'
        red, blue = [float(val) for val in self.awb_gains]
        s += 'awb gains: (red={:.2f}, blue={:.2f})\n'.format(red, blue)

        return s



        
    
        
