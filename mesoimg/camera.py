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
import queue
import numpy as np
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




class Camera:




    def __init__(self, **config):

        # Initialize wrapped picamera. Sets `_cam` attribute.
        self._init_cam()

        # Initialize frame attributes.
        self.frame_lock = Lock()
        self._frame = None
        self._frame_counter = 0
        self._frame_clock = Clock()

        # Still capture attributes.
        self.still_lock = Lock()
        self._still_info = {}

        # Streaming attributes.
        self.streaming_lock = Lock()
        self._streaming_info = {}

        # Threading and synchronization.
        self.frame_q = queue.Queue(maxsize=10)
        self.new_frame = Condition()


    #--------------------------------------------------------------------------#
    # Main camera settings.


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
    def channels(self, c: str) -> None:
        if c not in ('r', 'g', 'b', 'rgb'):
            raise ValueError(f"invalid channels '{ch}'")
        self._channels = c

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
    # various properties


    @property
    def frame(self):
        return self._frame

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
    def status(self) -> Dict[str, Any]:
        return {name : getattr(self, name) for name in STATUS_ATTRS}


    #--------------------------------------------------------------------------#
    # Public methods


    # Bulk attribute getting/setting

    def getattrs(self, keys: Iterable[str]) -> List:
        return [getattr(self, name) for name in keys]


    def setattrs(self, items: Dict[str, Any]) -> None:
        for key, val in items.items():
            setattr(self, key, val)


    # Opening/closing/reopening/resetting methods


    def close(self) -> None:
        """Clear attributes and close the camera."""
        self.clear()
        self._cam.close()


    def clear(self) -> None:
        self.clear_frame_attrs()
        self.clear_still_attrs()
        self.clear_streaming_attrs()


    def clear_frame_attrs(self) -> None:
        """
        Reset frame attributes.
        """
        with self.frame_lock:
            self._frame = None
            self._frame_counter = 0
            self._frame_clock = Clock()
            clear_queue(self.frame_queue)


    def clear_still_attrs(self) -> None:
        with self.still_lock:
            self._still_info = {}


    def clear_streaming_attrs(self) -> None:
        with self.streaming_lock:
            self._streaming_info = {}


    def reopen(self) -> None:
        """Close the camera if open, the reopen it."""
        self.close()
        self._init_cam()


    # Recording/previewing
    def capture(self, out: Optional[PathLike] = None) -> None:
        self.clear_frame_attrs()
        self.clear_still_attrs()


    def start_streaming(self) -> None:
        self.clear_frame_attrs()
        self.clear_streaming_attrs()
        buf = FrameBuffer(self)
        self._cam.start_recording(buf, 'rgb')


    def wait_streaming(self, timeout: float) -> None:
        self._cam.wait_recording(timeout)


    def stop_streaming(self) -> None:
        self._cam.stop_recording()
        time.sleep(0.1)



    #-------------------------------------------------------------------------#
    # Private/protected methods


    def _init_cam(self):

        # Start picamera, and update with requested config.
        self._cam = PiCamera(resolution=_DEFAULTS['resolution'],
                             framerate=_DEFAULTS['framerate'],
                             sensor_mode=_DEFAULTS['sensor_mode'])
        time.sleep(2.0)
        for key, val in _DEFAULTS.items():
            if key not in ('resolution', 'framerate', 'sensor_mode'):
                setattr(self, key, val)



    def _frame_callback(self, data: np.ndarray) -> None:
        pass

    def _recording_callback(self, data: np.ndarray) -> None:

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





class FrameBuffer(io.BytesIO):

    """
    Image buffer for unencoded RGB video.

    """

    #: Owning camera instance.
    _cam: Camera


    def __init__(self, cam: Camera):
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
            msg = f"Expected {self._n_bytes_in} bytes, received {bytes_available}"
            raise IOError(msg)

        # Reshape the data from the buffer, and send it to the camera.
        data = np.frombuffer(self.getvalue(), dtype=np.uint8)
        data = data.reshape(self._in_shape)[self._out_slice]
        data = as_contiguous(data)
        self._cam._frame_buffer_callback(data)

        # Finally, rewind the buffer and return as usual.
        self.truncate(0)
        self.seek(0)
        return n_bytes


    def flush(self) -> None:
        super().flush()


    def close(self) -> None:
        self.flush()
        self.truncate(0)
        self.seek(0)
        super().close()
