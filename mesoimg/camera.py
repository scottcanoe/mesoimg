import io
import logging
import os
from pathlib import Path
import re
from threading import Condition, Event, Lock, RLock, Thread
import time
from typing import (Any,
                    Callable,
                    ClassVar,
                    Dict,
                    List,
                    Iterable,
                    NamedTuple,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)
import queue
import numpy as np
import picamera
from picamera import PiCamera
import zmq
from mesoimg.buffers import *
from mesoimg.common import *
from mesoimg.timing import *
from mesoimg.outputs import *



__all__ = [
    'Camera',
]



_DEFAULTS: ClassVar[Dict[str, Any]] = {\
    'resolution' : (480, 480),
    'channels' : 'g',
    'framerate' :  30.0,
    'sensor_mode' : 7,
    'exposure_mode' : 'sports',
    'iso' : 0,
    'awb_mode' : 'off',
    'awb_gains' : (1.0, 1.0),
    'video_denoise' : False,
    }




_STATUS_ATTRS: ClassVar[Tuple[str]] = (\
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


def validate_channels(ch: str) -> str:
    ch = ch.lower()
    if ch not in ('r', 'g', 'b', 'rgb'):
        raise ValueError(f'invalid channel spec "{ch}"')
    return ch


class Camera:



    def __init__(self, **config):

        # Basic attributes.
        self.lock = Lock()

        # Frame attributes.
        self.frame_lock = Lock()
        self._frame = None
        self._frame_counter = 0
        self._frame_clock = Clock()

        # Threading and synchronization
        self.new_frame = Condition()
        self.frame_q = queue.Queue(maxsize=30)

        # Create the picamera instance with defaults.
        self._init_cam()


    #--------------------------------------------------------------------------#
    # Main camera settings.


    @property
    def resolution(self) -> Tuple[int, int]:
        res = self._cam.resolution
        return (int(res[0]), int(res[1]))

    @resolution.setter
    def resolution(self, res: Tuple[int, int]) -> None:
        if self.streaming:
            raise RuntimeError('cannot modify resolution while streaming.')
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
        if self.streaming:
            raise RuntimeError('cannot modify resolution while streaming.')
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
    # Extra properties


    @property
    def channels(self) -> str:
        return self._channels

    @channels.setter
    def channels(self, ch: str) -> None:
        if self.streaming:
            raise RuntimeError('cannot modify resolution while streaming.')
        self._channels = validate_channels(ch)

    @property
    def streaming(self) -> bool:
        """Whether the camera is busy recording data."""
        return self._cam.recording

    @property
    def frame(self) -> Optional[Frame]:
        """Most recently captured frame."""
        return self._frame

    @property
    def out_shape(self) -> Tuple[int]:
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
    def status(self) -> Dict[str, Any]:
        """Dictionary containing various properties."""
        return {name : getattr(self, name) for name in STATUS_ATTRS}


    #--------------------------------------------------------------------------#
    # Opening, closing, etc.

    def _init_cam(self):

        """

        """

        from mesoimg.app import kill_zombied_camera, write_snippet

        resolution  = _DEFAULTS['resolution']
        framerate   = _DEFAULTS['framerate']
        sensor_mode = _DEFAULTS['sensor_mode']

        with self.lock:

            # Start picamera, and update with requested config.
            try:
                self._cam = PiCamera(resolution=resolution,
                                     framerate=framerate,
                                     sensor_mode=sensor_mode)
            except picamera.exc.PiCameraMMALError:
                kill_zombied_camera()
                self._cam = PiCamera(resolution=resolution,
                                     framerate=framerate,
                                     sensor_mode=sensor_mode)

            write_snippet('picamera.pid', str(os.getpid()))

            for key, val in _DEFAULTS.items():
                if key not in ('resolution', 'framerate', 'sensor_mode'):
                    setattr(self, key, val)

            time.sleep(1.0)


    def clear(self) -> None:

        if self.streaming:
            raise RuntimeError("Cannot clear camera while streaming. ")

        with self.frame_lock:
            self._frame = None
            self._frame_counter = 0
            self._frame_clock = Clock()
            clear_q(self.frame_q)


    def close(self) -> None:
        """Close the picamera instance."""
        self._cam.close()


    def reopen(self) -> None:
        """
        Close the current PiCamera instance, and replace it
        with a newly initialized one. Also resets various attributes,
        (e.g., ``frame``, ``frame_q``).
        """
        self.close()
        self._init_cam()
        self.clear()


    #--------------------------------------------------------------------------#
    # Capturing and streaming methods


    def capture(self,
                out: Optional[PathLike] = None,
                use_video_port: bool = True,
                ) -> None:

        if self.streaming:
            raise RuntimeError('Cannot capture while streaming in progress.')

        self.clear()
        buf = FrameBuffer(self)
        self._cam.capture(buf, 'rgb', use_video_port=True)
        if out:
            fn = get_writer(out)
            fn(out, self.frame.data)


    def start_streaming(self) -> None:

        if self.streaming:
            raise RuntimeError('Camera is already streaming.')

        self.clear()
        buf = FrameBuffer(self)
        self._cam.start_recording(buf, 'rgb')


    def wait_streaming(self, timeout: float) -> None:
        if not self.streaming:
            raise RuntimeError('Camera is not streaming.')
        self._cam.wait_recording(timeout)


    def stop_streaming(self) -> None:
        if not self.streaming:
            raise RuntimeError('Camera is not streaming.')
        self._cam.stop_recording()
        time.sleep(0.01)


    def stream_for(self, duration: float) -> None:
        self.start_streaming()
        self.wait_streaming(duration)
        self.stop_streaming()


    def stream_until(self, event: Any) -> None:
        pass


    #--------------------------------------------------------------------------#
    # utilities


    def getattrs(self, keys: Iterable[str]) -> List:
        return [getattr(self, name) for name in keys]


    def setattrs(self, items: Dict[str, Any]) -> None:
        for key, val in items.items():
            setattr(self, key, val)


    #-------------------------------------------------------------------------#
    # Private/protected methods


    def _frame_callback(self, data: np.ndarray) -> None:

        """
        Called by the frame buffer upon new frame being written.
        """

        # Update frame and its counters.

        with self.frame_lock:
            index, timestamp = self._frame_counter, self._frame_clock()
            self._frame_counter += 1
            self._frame = Frame(data=data, index=index, timestamp=timestamp)

        # Notify.
        with self.new_frame:
            put_q(self.frame_q, self._frame)
            self.new_frame.notify_all()


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
        reprt = f'Produced {n_frames} frames in {duration} secs. (fps={fps})'
        print(report, flush=True)


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



