import io
import logging
import os
from pathlib import Path
import re
from threading import Condition, Event, Lock, Thread
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
import zmq

import picamera
from mesoimg.common import *
from mesoimg.timing import *
from mesoimg.outputs import Frame, FrameBuffer, FrameStream, H5WriteStream



__all__ = [
    'Camera',
]

      
        
class Camera:
    
    
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
                 framerate: float = 15.0,
                 sensor_mode: int = 7,
                 awb_mode: str = 'off',
                 awb_gains: Tuple[float, float] = (1.0, 1.0),
                 cmd_sock = None,
                 frame_sock = None,
                 status_sock = None,
                 ):

                        
        # Events, flags, locks.
        self.lock = Lock()
        self._frame = None
        self._index = 0
        self._clock = Clock()
        self._timestamp = self._clock()
        
        self._active = False
        self._countdown_timer = None
        self._stop = False
        self.stop_event = Event()
        
        # Attributes.
        self.channels = channels

        # Networking attributes.
        self.cmd_sock = cmd_sock
        self.frame_sock = frame_sock
        self.status_sock = status_sock
                                                    
        # The picamera instance.
        print('Initializing camera.', flush=True)
        self._cam = picamera.PiCamera(resolution=resolution,
                                      framerate=framerate,
                                      sensor_mode=sensor_mode)
                                      
        time.sleep(1.0)
        self.awb_mode = awb_mode
        self.awb_gains = awb_gains
        time.sleep(1.0)
        self.stash_attrs()
                                
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
    def index(self) -> int:
        return self._index

    @property
    def timestamp(self) -> float:
        return self._timestamp
    
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
    def status(self):
        return self.getstatus()
        
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

    def reset(self):
        print('Resetting camera.', flush=True)
        with self.lock:
            self._active = False
            self._frame = None
            self._frame_counter = 0
            self._index = 0
            self._timestamp = 0
            if self._clock is not master_clock:
                self._clock.reset()
            self._countdown_timer = None
            self._stop = False
        print('Done.', flush=True)
    
    
    def start_preview(self):

        self.reset()
        self.frame_buffer = FrameBuffer(self)
        with self.lock:
            self._active = True
            self._countdown_timer = None
        print(f'Starting preview.', flush=True)
        self._cam.start_recording(self.frame_buffer, 'rgb')
        while not self._stop:
            self._cam.wait_recording(1)
        self._cam.stop_recording()
        print(f'Previewed {self._index} frames')

        
    def stop_preview(self):
        self._cam.stop_recording()
        
    
    def start_recording(self,
                        duration: float,
                        interval: float = 2.0,
                        ) -> None:

        
        """
        Record for a fixed period of time.
        """
        
        self.reset()
        # Setup frame buffer.
        self.frame_buffer = FrameBuffer(self)
        with self.lock:
            self._active = True
            self._countdown_timer = CountdownTimer(duration)
        print(f'Starting recording: {duration} secs.', flush=True)
        self._cam.start_recording(self.frame_buffer, 'rgb')
        while True:
            self._cam.wait_recording(interval)
            if self._stop or self._countdown_timer() <= 0:
                break
            
        self._cam.stop_recording()
    
    
    def stop_recording(self) -> None:
        print(f'Stopping recording.', flush=True)
        #with self.lock:
        self._stop = True
        self._active = False
        self._cam.stop_recording()
        print('Done.', flush=True)
        
    
    def _write_callback(self, data: np.ndarray):

        self._index = self._frame_counter
        self._frame_counter += 1
        self._timestamp = self._clock()
        if self._index % 30 == 0:
            print(f'Frame: {self._index}', flush=True)            
            
        self._frame = Frame(data=data,
                            index=self._index,
                            timestamp=self._timestamp)

        if self.frame_sock:
            frame = self._frame
            md = {'shape': frame.data.shape,
                  'dtype': str(frame.data.dtype),
                  'index': frame.index,
                  'timestamp' : frame.timestamp}
            self.frame_sock.send_json(md, zmq.SNDMORE)
            self.frame_sock.send(frame.data.tobytes(),
                                 0, # flags
                                 copy=True,
                                 track=False)
            time.sleep(0.005)
#            msg = self.frame_sock.recv_string()
#            if msg == 'ready':
#                send_frame(self.frame_sock, self._frame)
#            else:
#                self._stop = True
                    
        #if self._countdown_timer and self._countdown_timer() <= 0:
         #   self._stop = True
        
                
                
    def _write_callback2(self, data: np.ndarray) -> None:

        if self._stop:
            self.stop_recording()
            return

        with self.lock:
            self._index = self._frame_counter
            self._frame_counter += 1
            self._timestamp = self._clock()

            print(f'Frame: {self._index}', flush=True)            
#            if index % 30 == 0:
#                print(f'Frame: {index}', flush=True)

                
            self._frame = Frame(data=data,
                                index=self._index,
                                timestamp=self._timestamp)

            # Send frame.
            if self.frame_sock:
                msg = self.frame_sock.recv_string()
                if msg == 'ready':
                    send_frame(self.frame_sock, self._frame)
                elif msg == 'stop':
                    self._stop = True

            # Send status.
            if self.status_sock:
                msg = self.status_sock.recv_string()
                if msg == 'ready':
                    stat = self.getstatus()
                    self.status_sock.send_json(stat)
                elif msg == 'stop':
                    self._stop = True
                    
            if self._countdown_timer and self._countdown_timer.clock() <= 0:
                self._stop = True
        # Stop recording if requested.

        if self._stop:
            self.stop_recording()
            return                

    #-----------------------------------------------------------#
    # Housekeeping


    def stash_attrs(self) -> Dict:
        """Stash current attributes into self.attrs"""
        self.stash = {key : getattr(self, key) for key in self.ATTRIBUTES}
        return self.stash


    def restore_attrs(self) -> None:
        self.setattrs(self.stash)
    
                                       
    def close(self) -> None:

        if not self.closed:
            self.stash_attrs()
            self._cam.close()
            self._cam = None

            
    def reopen(self) -> None:
        if not self.closed:
            self.close()

        resolution  = self.stash['resolution']
        framerate   = self.stash['framerate']
        sensor_mode = self.stash['sensor_mode']
        self._cam = picamera.PiCamera(resolution=resolution,
                                      framerate=framerate,
                                      sensor_mode=sensor_mode)
        time.sleep(2)
        
    #-----------------------------------------------------------#
    # Networking

                                                
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


def send_frame(socket,
               frame: Frame,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> None:

    md = {'shape': frame.data.shape,
          'dtype': str(frame.data.dtype),
          'index': frame.index,
          'timestamp' : frame.timestamp}
    socket.send_json(md, flags | zmq.SNDMORE)
    socket.send(frame.data.tobytes(), flags, copy=copy, track=track)  
    
            
