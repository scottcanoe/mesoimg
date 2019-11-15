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
import matplotlib.pyplot as plt
import numpy as np
import picamera
from mesoimg.common import *
from mesoimg.timing import Clock, IntervalTimer
from mesoimg.outputs import Frame, FrameBuffer, FrameStream, H5WriteStream



__all__ = [
    'Camera',
]




        
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


    recording: Event
    clock: Clock
    channels: Channels
    frame_buffer: Optional['FrameBuffer']
    
    
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
                         
        self.recording = Event()
        self.clock = Clock()
        self.channels = channels
        self.frame_buffer = None
        
        if sleep:
            time.sleep(sleep)
            
        
    @property
    def channels(self) -> Channels:
        return self._channels
            
    @channels.setter
    def channels(self, ch: Union[str, Channels]) -> None:

        if self.recording.is_set():
            msg  = "Cannot modify 'channels' attribute "
            msg += "while recording/previewing."
            raise RuntimeError(msg)
        self._channels = Channels(ch)

        
                
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
                
        
    def _new_frame_callback(self, frame: Frame) -> None:
    
        """
        Called by FrameBuffer when a new frame is available.
        """                        
        pass
    
    
    def preview(self):
        """
        Stream video to an animated figure, albeit slowly (3-4 frames/sec.).
        Without 
        """
        
        from mesoimg.display import ImageViewer
        
        print("Starting preview", flush=True)

        try:


            viewer = ImageViewer(self)
            self.frame_buffer = FrameBuffer(self)
            self.recording.set()
            for _ in self.capture_continuous(self.frame_buffer,
                                            'rgb',
                                             use_video_port=True):
                frame = self.frame_buffer.frame
                viewer.update(frame)
                if viewer.closed:
                    break

        except Exception as exc:

            if not exc.__class__.__name__ == 'TclError':
                self.recording.clear()
                self.frame_buffer.flush()
                self.frame_buffer.close()
                self.close()
                raise
        
        self.recording.clear()
        self.frame_buffer.flush()
        self.frame_buffer.close()     
        
        print("Preview finished.", flush=True)

        



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



        
    
        
