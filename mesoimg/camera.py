import io
import logging
import os
from pathlib import Path
import re
from threading import Event, Lock, Thread
import time
from typing import (Any,
                    Callable,
                    ClassVar,
                    Dict,
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
        If > 0, will used time.sleep() to allow camera to warm up it's sensors..

    """

    COMMAND_PORT = 7000
    DATA_PORT    = 7001
    STATUS_PORT  = 7002
    
    ATTRIBUTES = [
        # hardware
        'resolution',
        'channels',
        'framerate',
        'sensor_mode',
        'exposure_mode',
        'analog_gain',
        'digital_gain',
        'iso',
        'exposure_speed',
        'shutter_speed',
        'awb_mode',
        'awb_gains',
        'closed',
        # capture metadata
        'index',
        'timestamp',
    ]
    
    
    def __init__(self,
                 resolution: Tuple[int, int] = (640, 480),
                 channels: str = 'g',
                 framerate: float = 30.0,
                 sensor_mode: int = 7,
                 networking: bool = False,
                 ):

                        
        # Events, flags, locks.
        self._lock = Lock()
        self._frame = None
        self._index = 0
        self._clock = Clock()
        self._timestamp = self._clock()

        self._recording = False
        self.write_complete = False
        self.abort = False
        self.stop_event_loop = False
                                        
        # Attributes.
        self.channels = channels

        # Networking attributes.           
        self._networking = networking        
        self.zmq_context = None
        self.cmd_sock    = None
        self.data_sock   = None
        self.stat_sock   = None
        if self._networking:
            self.open_sockets()
                                                    
        # The picamera instance.
        print('Initializing camera.', flush=True)
        self._cam = picamera.PiCamera(resolution=resolution,
                                      framerate=framerate,
                                      sensor_mode=sensor_mode)
                                      
        time.sleep(2.0)        
        self.stache_attrs()
                                
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
    def channels(self) -> str:
        return self._channels
            
    @channels.setter
    def channels(self, ch: str) -> None:
        if ch not in ('r', 'g', 'b', 'rgb'):
            raise ValueError(f"invalid channels '{ch}'")
        self._channels = ch

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
    def frame(self) -> Frame:
        return self._frame

    @property
    def recording(self) -> bool:
        return self._recording        
    
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
                

    @property
    def networking(self) -> bool:
        return self._networking


    #-----------------------------------------------------------#
    # Get/Set methods.


    def getattrs(self, attrs: Sequence[str]) -> Dict:
        """Bulk attribute getting"""
        return {name : getattr(self, name) for name in attrs}

    def getstatus(self) -> Dict:
        return {name : getattr(self, name) for name in self.ATTRIBUTES}
    
    def setattrs(self, data: Dict[str, Any]) -> None:
        """Bulk attribute setting"""
        for key, val in data.items():
            setattr(self, key, val)


    #-----------------------------------------------------------#
    # Recording/streaming methods.

    def record(self,
               outfile: PathLike,
               duration: float,
               overwrite: bool = False,
               ) -> FrameBuffer:
        
        """
        Record for a fixed period of time.
        """
        
        if self._recording or self.write_complete or self.abort:
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
        with self._lock:
            out.close()
            outfile.close()
            self.write_complete = False
            self.abort = False

        
                
    def start_recording(self) -> FrameBuffer:

        frame_buffer = FrameBuffer(self)

        with self._lock:
            self._recording = True
            self._frame_counter = 0
            self._index = 0
            self._clock.reset()
            self._timestamp = 0
            
        print('Starting recording', flush=True)
        self._cam.start_recording(frame_buffer, 'rgb')
        return out
        
       
    def wait_recording(self, timeout: float = 0) -> None:
        """
        Poll for events/handle user intervention while
        waiting for recording to finish.
        """

        t_stop = master_clock() + timeout
        self._cam.wait_recording(timeout)
        
        
    def stop_recording(self) -> None:

        self._cam.stop_recording()
        with self._lock:
            self._recording = False
        print('Recording stopped.', flush=True)
        

    def _write_callback(self, data: np.ndarray) -> None:

        with self._lock:
            self._frame = Frame(data=data,
                                index=self._frame_counter,
                                timestamp=self._timestamp)
            self._frame_counter += 1
            self._index = self._frame.index
            self._timestamp = self._frame.timestamp
        

        
        if read_stdin().strip() == 'q':
            print('user abort', flush=True)
            with self.lock:
                self.abort = True

    #-----------------------------------------------------------#
    # Housekeeping


    def stash_attrs(self) -> Dict:
        """Stash current attributes into self.attrs"""
        self.stash = {key : getattr(self, key) for key in self.ATTRIBUTES}
        return self.stash


    def restore_attrs(self) -> None:
        self.setattrs(self.stache)
    
    
    def cleanup(self) -> None:
        with self._lock:
            self._recording = False
            
                        
    def close(self) -> None:
        
        self.cleanup()
        if not self.closed:
            self.stash_attrs()
            self._cam.close()
            self._cam = None

            
    def reopen(self) -> None:
        if not self.closed:
            return

        resolution  = self.stache['resolution']
        channels    = self.stache['channels']
        framerate   = self.stache['framerate']
        sensor_mode = self.stache['sensor_mode']
        print('Reopening camera')
        self._cam = picamera.PiCamera(resolution=resolution,
                                      channels=channels,
                                      framerate=framerate,
                                      sensor_mode=sensor_mode)
        time.sleep(2)
        
    #-----------------------------------------------------------#
    # Networking

    def open_sockets(self):

        self.zmq_context = zmq.Context()

        # Open the command port for receiving client commands.       
        self.cmd_sock = self.context.socket(zmq.REP)
        self.cmd_sock.bind(f'tcp://*:{self.COMMAND_PORT}')

        # Open the data port for streaming data to subscribers.
        self.data_sock = self.context.socket(zmq.PUB)
        self.data_sock.bind(f'tcp://*:{self.DATA_PORT}')
    
        self.stat_sock = self.context.socket(zmq.PUB)
        self.stat_sock.bind(f'tcp://*:{self.STATUS_PORT}')
    
        
    def close_sockets(self):

        # Close/terminate all sockets and zmq context.
        self.cmd_sock.close()
        self.data_sock.close()
        self.stat_sock.close()
        self.zmq_context.term()

                
                                                
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
            s += ' (gain locked at current values)'
        s += '\n'
        
        s += f'analog gain: {float(self.analog_gain)}\n'
        s += f'digital gain: {float(self.digital_gain)}\n'
        s += f'ISO : {self.iso}\n'
        s += f'exposure speed: {self.exposure_speed} microsec.\n'
        s += f'shutter speed: {self.shutter_speed} microsec.\n'
        

        # Report auto-white balance.
        s += f'awb mode: {self.awb_mode}\n'
        red, blue = [float(val) for val in self.awb_gains]
        s += 'awb gains: (red={:.2f}, blue={:.2f})\n'.format(red, blue)

        return s



    
        
