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
                    List,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)
from queue import Queue
import numpy as np
import zmq

import picamera
from mesoimg.common import *
from mesoimg.timing import *
from mesoimg.outputs import *



__all__ = [
    'Camera',
]



class Camera:



    _STATUS_ATTRS: ClassVar[Tuple[str]] = (\
        'state',
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
        'video_denoise',
        )


    _DEFAULT_CONFIG: ClassVar[Dict[str, Any]] = {\
        'resolution' : (480, 480),
        'channels' : 'g',
        'framerate' :  20.0,
        'sensor_mode' : 7,
        'exposure_mode' : 'sports',
        'iso' : 0,
        'awb_mode' : 'off',
        'awb_gains' : (1.0, 1.0),
        'video_denoise' : False,
        }


    #: Configurations. Always includes 'default' and 'init'.
    _configs: Dict[str, Dict]

    #: Wrapped PiCamera instance. Gets set in ``Camera.init_cam``
    _cam: Optional['PiCamera'] = None

    #: Current state of camera. One of 'waiting', 'recording', 'previewing'.
    _state: str = 'waiting'

    #: Index of current frame being captured.
    _frame_counter: int = 0

    #: Most recent frame captured.
    _frame: Optional[Frame] = None

    #: Clock used for timestamps.
    _clock: Optional[Clock] = None


    def __init__(self, **config):

        self._configs = {}
        self._configs['default'] = self._DEFAULT_CONFIG.copy()
        self.init_cam(config)

        self.lock = Lock()
        self.frame = None
        self.new_frame = Condition()

        self._frame_buffer = FrameBuffer(self)
        self._frame_counter = 0
        self._clock = Clock()
        self._countdown_timer = None
        self._state = 'waiting'


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
    def video_denoise(self) -> bool:
        return self._cam.video_denoise

    @video_denoise.setter
    def video_denoise(self, val: bool) -> None:
        self._cam.video_denoise = val

    @property
    def closed(self):
        return self._cam.closed


    #--------------------------------------------------------------------------#
    # etc. properties


    @property
    def state(self):
        return self._state


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



    #--------------------------------------------------------------------------#
    # Configuration  getting/setting and status reporting

    @property
    def config(self) -> Dict[str, Any]:
        attrs = sorted(list(self._DEFAULT_CONFIG.keys()))
        return {name : getattr(self, name) for name in attrs}


    @config.setter
    def config(self, c: Union[str, Dict]) -> None:
        if isinstance(c, str):
            c = self._configs[c]
        for key, val in c.items():
            setattr(self, key, val)


    def stash_config(self, name: str = 'stash') -> None:
        """Stash current attributes into self.attrs"""
        self._configs[name] = self.config


    @property
    def status(self) -> Dict[str, Any]:
        return {name : getattr(self, name) for name in self._STATUS_ATTRS}


    #--------------------------------------------------------------------------#
    # Opening/closing/reopening/resetting methods


    def init_cam(self, config: Dict[str, Any]) -> None:
        """
        Sets self._cam and self._init_config.
        """
        if self._cam is not None:
            print('PiCamera instance already exists. Doing nothing', flush=True)
            return

        # Fill in any missing configurations with defaults.
        config = config.copy()
        for key, val in self._DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = val

        # Create the picamera instance, and set desired configuration.
        print('Initializing camera.', flush=True)
        self._cam = picamera.PiCamera(resolution=config['resolution'],
                                      framerate=config['framerate'],
                                      sensor_mode=config['sensor_mode'])
        time.sleep(2.0)
        for key, val in config.items():
            if key in ('resolution', 'framerate', 'sensor_mode'):
                continue
            setattr(self, key, val)

        self._configs['init'] = config


    def close(self) -> None:
        try:
            if not self.closed:
                print('Closing camera.', flush=True)
                self.stash_config('last')
                self._cam.close()
                time.sleep(0.1)
        except:
            try:
                self._cam.close()
            except:
                pass


    def reopen(self, config: Union[str, Dict] = 'init') -> None:

        if not self.closed:
            self.close()

        if isinstance(config, str):
            try:
                config = self._configs[config]
            except KeyError:
                print(f'No such config: {config}. Falling back to default.')
                config = self._DEFAULT_CONFIG
        self.init_cam(config)


    def _reset(self):

        # while not self.frame_queue.empty():
        #     self.frame_queue.get()

        # Reset frame and its counters.
        with self.lock:
            self.frame = None
            self._frame_counter = 0
            if self._clock is not master_clock:
                self._clock.reset()

        # Reinitialize frame buffer.
        if self._frame_buffer:
            self._frame_buffer.close()
            self._frame_buffer = None
            time.sleep(0.01)
        self._frame_buffer = FrameBuffer(self)

        # Clear remaining state variables.
        self._countdown_timer = None
        self._state = 'waiting'



    #--------------------------------------------------------------------------#
    # Recording/previewing


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



    #-------------------------------------------------------------------------#
    # Private/protected methods


    def _write_callback(self, data: np.ndarray) -> None:

        """
        Called by the frame buffer upon new frame being written.
        """

        # Update frame and its counters.
        with self.lock:
            index, timestamp = self._frame_counter, self._clock()
            self.frame = Frame(data=data,
                               index=index,
                               timestamp=timestamp)
            self._frame_counter += 1

        # Notify frame emitters.
        with self.new_frame:
            self.new_frame.notify_all()

        # Report.
        if index % 30 == 0:
            print(f'Frame: {index}', flush=True)


    def test(self, duration: float = 2.0, interval: float = 1.0) -> None:

        self._reset()
        self._countdown_timer = CountdownTimer(duration)

        self._cam.start_recording(self._frame_buffer, 'rgb')
        while True:
            self._cam.wait_recording(interval)
            if self._countdown_timer() <= 0:
                break
        self._cam.stop_recording()

        n_frames = self._frame_counter - 1
        fps = n_frames / duration
        print(f'Produced {n_frames} frames in {duration} secs. (fps={fps})', flush=True)


    def __repr__(self):

        if self.closed:
            return 'Camera (closed)'

        s  = '       Camera      \n'
        s += '-------------------\n'
        s += f'sensor mode: {self.sensor_mode}\n'
        s += f'resolution: {self.resolution}\n'
        s += f'channels: {str(self.channels)}\n'
        s += f'"framerate": {self.framerate}\n'

        s += f'exposure mode: {self.exposure_mode}'
        if self.exposure_mode == 'off':
            s += ' (gain locked at current values)'
        s += '\n'

        s += f'analog gain: {float(self.analog_gain)}\n'
        s += f'digital gain: {float(self.digital_gain)}\n'
        s += f'iso : {self.iso}\n'
        s += f'exposure speed: {self.exposure_speed} usec.\n'
        s += f'shutter speed: {self.shutter_speed} usec.\n'


        # Report auto-white balance.
        s += f'awb mode: {self.awb_mode}\n'
        red, blue = [float(val) for val in self.awb_gains]
        s += 'awb gains: (red={:.2f}, blue={:.2f})\n'.format(red, blue)

        return s





