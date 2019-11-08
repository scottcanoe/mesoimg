"""
Script used to figure out how to stream video and get timestamps for each frame.
"""
import io
import logging
from pathlib import Path
import sys
import time
from typing import Callable, Optional, Tuple, Union
import warnings
import numpy as np
from .common import PathLike, uint8, Clock



__all__ = [
    'Camera',
]

    
    
class CameraFlags:
    
    def __init__(self, parent: 'Camera'):
        self._parent = parent
        self.reset()
    
    @property
    def CAPTURING(self):
        return self._CAPTURING
    
    @property
    def RECORDING(self):
        return self._RECORDING
   
    @property
    def INTERRUPT(self):
        return self._INTERRUPT

    def reset(self):
        self._CAPTURING = False
        self._RECORDING = False
        self._INTERRUPT = False
        
    
    

class Camera:
    
    """
    Wrapper for picamera's PiCamera class tailored towards one-photon mesoscale 
    GCaMP imaging.
    
    
    Data Formats
    ============
    
    
    
    
    Sensor Modes
    ============
    
    On the V2 camera module, the sensor modes are:

    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    | # | Resolution | Aspect Ratio | Framerates        | Video | Image | FoV     | Binning |
    +===+============+==============+===================+=======+=======+=========+=========+
    | 1 | 1920x1080  | 16:9         | 1/10 <= fps <= 30 | x     |       | Partial | None    |
    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    | 2 | 3280x2464  | 4:3          | 1/10 <= fps <= 15 | x     | x     | Full    | None    |
    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    | 3 | 3280x2464  | 4:3          | 1/10 <= fps <= 15 | x     | x     | Full    | None    |
    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    | 4 | 1640x1232  | 4:3          | 1/10 <= fps <= 40 | x     |       | Full    | 2x2     |
    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    | 5 | 1640x922   | 16:9         | 1/10 <= fps <= 40 | x     |       | Full    | 2x2     |
    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    | 6 | 1280x720   | 16:9         | 40 < fps <= 90    | x     |       | Partial | 2x2     |
    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    | 7 | 640x480    | 4:3          | 40 < fps <= 90    | x     |       | Partial | 2x2     |
    +---+------------+--------------+-------------------+-------+-------+---------+---------+
    
    Most likely, mode 7 will be the most useful for our application.
    

    Fixing Gain and Exposure Times
    ==============================
    
    Probably will want to let gains settle, then lock them by setting `exposure_mode` to 
    'off'. According to picamera:
    
    Setting exposure_mode to 'off' locks the analog (and digital) gains at their current
    values and doesn’t allow them to adjust at all, no matter what happens to the scene,
    and no matter what other camera attributes may be adjusted.

    Setting exposure_mode to values other than 'off' permits the gains to “float” (change)
    according to the auto-exposure mode selected. Where possible, the camera firmware
    prefers to adjust the analog gain rather than the digital gain, because increasing
    the digital gain produces more noise. Some examples of the adjustments made for
    different auto-exposure modes include:
        - 'sports' reduces motion blur by preferentially increasing gain rather than
        exposure time (i.e. line read-out time).
        - 'night' is intended as a stills mode, so it permits very long exposure times
        while attempting to keep gains low.

    The iso attribute effectively represents another set of auto-exposure modes with
    specific gains:
        - With the V1 camera module, ISO 100 attempts to use an overall gain of 1.0.
        ISO 200 attempts to use an overall gain of 2.0, and so on.
        - With the V2 camera module, ISO 100 produces an overall gain of ~1.84.
        ISO 60 produces overall gain of 1.0, and ISO 800 of 14.72 (the V2 camera module
        was calibrated against the ISO film speed standard).

    Hence, one might be tempted to think that iso provides a means of fixing the gains,
    but this isn’t entirely true: the exposure_mode setting takes precedence
    (setting the exposure mode to 'off' will fix the gains no matter what ISO is later
    set, and some exposure modes like 'spotlight' also override ISO-adjusted gains).
    
    
    

    """
    
    
    # Image attributes.
    _channels = 'rgb'         # Which channels to keep (e.g., 'g' for just green).
    _format = 'rgb'           # May be h264, 'rgb', etc.
    _gain = None
    
    # Video attributes.
    # ...
    
    # Dependent/computed attributes.
    _bytes_per_frame = None
    _channel_indices = None
    _frame_shape = None
    _n_channels = None
    
    # Wrapped objects, flags, etc.
    _cam = None
    _flags = None

    
    def __init__(self,
                 resolution: Union[None, Tuple[int, int]] = None,
                 frame_rate: Union[None, int] = None,
                 sensor_mode: int = 0,
                 format: str = 'rgb',
                 channels: str = 'rgb'):
        
        import picamera

        self._cam = picamera.PiCamera(resolution=resolution,
                                      framerate=frame_rate,
                                      sensor_mode=sensor_mode)
        time.sleep(2)
        self._flags = CameraFlags(self)
        self.format = format
        self.channels = channels
        self._update_attrs()
        
        

    #-------------------------------------------------------------------------------------
    # Image properties. 
    
    @property
    def channels(self) -> str:
        """String that represents which rgb channels are to be kept. 
        For example, all rgb channels will be kept (in that order) by setting
        this to 'rgb'. Keeping only the green channel is done by setting channels to 'g'.
        Note that the order matters: 'rgb' is not the same as 'bgr'.
        """
        return self._channels
    
    @channels.setter
    def channels(self, ch: str) -> None:
        if not all(c in 'rgb' for c in ch):
            raise ValueError("Invalid channels string'{}'.".format(ch)) 
        self._channels = ''.join(ch)
        self._update_attrs()

    @property
    def format(self) -> str:
        return self._format
    
    @format.setter
    def format(self, fmt: str) -> None:
        self._format = fmt
                        
    @property
    def resolution(self) -> Tuple[int, int]:
        """Camera's resolution settings.
        """
        return self._cam.resolution
        
    @resolution.setter
    def resolution(self, res: Tuple[int, int]) -> None:
        self._cam.resolution = res
        self._update_attrs()
        
    @property
    def sensor_mode(self) -> int:
        return self._cam.sensor_mode
    
    @sensor_mode.setter
    def sensor_mode(self, mode: int) -> None:
        self._cam.sensor_mode = mode
        self._update_attrs()

   
    #-------------------------------------------------------------------------------------
    # Video properties. 

    @property
    def frame_rate(self) -> float:
        """Desired frame rate.
        """
        return self._cam.framerate
        
    @frame_rate.setter
    def frame_rate(self, fs: float) -> None:
        self._cam.framerate = fs

    #-------------------------------------------------------------------------------------
    # Dependent/computed properties. 

    @property
    def bytes_per_frame(self):
        """Computed property provided for convenience."""
        return self._bytes_per_frame

    @property
    def frame_shape(self):
        """Shape of captured frames. May be 2d (single-channel) or 3d (multi-channel)."""
        return self._frame_shape
    
    @property
    def n_channels(self):
        """Number of channels to capture."""
        return self._n_channels


    #-------------------------------------------------------------------------------------
    # Wrapped properties and objects.

    @property
    def cam(self):
        """Wrapped picamera.PiCamera."""
        return self._cam

    @property
    def closed(self) -> bool:
        """Wraps picamera.PiCamera's `closed` read-only property."""
        return self._cam.closed
   
    @property 
    def flags(self):
        """Camera flags."""
        return self._flags
   
   
    #------------------------------------------------------------------------------------#
    # Image capture
        
    def capture(self, out, **kw):
        
        # Alias.
        flags = self._flags
        bytes_per_frame = self._bytes_per_frame
        fmt = self.format if format is None else format
        
        # Make sure we're not already capturing an image or streaming frames.
        self._check_can_start()
        flags._RECORDING = True



    #------------------------------------------------------------------------------------#
    # Video capture

    def record(self):
        pass

    def start_recording(self, *args, **kw):
        raise NotImplementedError
        
    def wait_recording(self, *args, **kw):
        raise NotImplementedError

    def stop_recording(self, *args, **kw):
        raise NotImplementedError


    def iterframes(self,
                   duration: Optional[float] = None,
                   n_frames: Optional[int] = None,
                   format: Optional[str] = None,
                   use_video_port: bool = True,
                   callback: Optional[Callable] = None) -> 'ImagingFrame':
        
        """
        Yield imaging frames iteratively, stopping when either camera.stop() is called,
        or `duration` is given and the time has expired, or `n_frames` was given
        and the frame counter has expired (whichever comes first).
        
        """

        # Alias.
        flags = self._flags
        bytes_per_frame = self._bytes_per_frame
        fmt = self.format if format is None else format
        
        # Make sure we're not already capturing an image or streaming frames.
        self._check_can_start()
        flags._RECORDING = True
        
        # Initialize output stream.
        stream = io.BytesIO()

        # Yield frames.
        frame_count = 0
        clock = Clock(start=True)
        stream = '/home/pi/mov.h264'
        fmt = 'h264'
        for buf in self.cam.capture_continuous(stream, fmt, use_video_port):

            # Check for termination conditions.
            frame_count += 1
            timestamp = clock.time()
            print('timestamps: {}'.format(timestamp), flush=True)
            
            if frame_count > n_frames:
                break
            yield None
            #if flags.INTERRUPT or \
               #(n_frames is not None and frame_count > n_frames) or \
               #(duration is not None and timestamp > duration):
                #break

            #stream.seek(0)
            #yield None
            # Get and check frame data.
            #data = stream.getvalue()
            #p, q = np.divmod(len(data), bytes_per_frame)
            #if p == 0:
                #raise IOError('no frames in buffer.')
            #elif p > 1:
                #raise IOError('more than one frame in buffer.')
            #if q != 0:
                #raise IOError('partial frames encountered in buffer.')            
        
            #stream.truncate(0)
            #stream.seek(0)
            #return data
            #data = self.proccess_frame_array(data)
            #frame = ImagingFrame(self, data, timestamp)
            #yield data
            
        
            


    def stop(self) -> None:
        """Interrupt capture."""
        if self._flags.CAPTURING or self._flags.RECORDING:
            self._flags._INTERRUPT = True
        else:
            logging.info('Nothing to interrupt.')
        
        
    #------------------------------------------------------------------------------------#
    # etc.
    
    def close(self):
        """Close the camera.
        """
        self._cam.close()


    def proccess_frame_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Reshape flat frame data into ndarray, and remove unwanted channels.
        
        """
        data = np.frombuffer(arr, dtype=uint8) if isinstance(arr, bytes) else data
        data = data.reshape([self._frame_shape[0], self._frame_shape[1], 3])
        data = data[:, :, self._channel_indices]
        return data


    def _check_can_start(self):
        if self._flags.CAPTURING or self._flags.RECORDING:
            raise RuntimeError('Already capturing/recording data.')
    
    
    def _update_attrs(self):
        """Recompute dependent attributes (e.g., ``bytes_per_frame``).
        """

        # Handle channel info.
        self._n_channels = len(self.channels)
        self._channel_indices = np.array(['rgb'.find(c) for c in self._channels])
        
        # Determine bytes per frame and frame shape.
        n_xpix, n_ypix = self.resolution
        self._bytes_per_frame = n_ypix * n_xpix * self._n_channels
        if self._n_channels == 1:
            self._frame_shape = (n_ypix, n_xpix)
        else:
            self._frame_shape = (n_ypix, n_xpix, self._n_channels)
            
        
    def __repr__(self):
        s  = '         Camera        \n'
        s += '-----------------------\n'
        s += 'resolution: {}\n'.format(self.resolution)
        s += 'channels: {}\n'.format(self.channels)
        s += 'format: {}\n'.format(self.format)
        s += 'sensor mode: {}\n'.format(self.sensor_mode)
        s += 'frame rate: {}\n'.format(self.frame_rate)
        s += 'frame shape: {}\n'.format(self.frame_shape)
        s += 'closed: {}\n'.format(self.closed)
        return s



class ImagingFrame:
    
    
    def __init__(self,
                 parent: Camera,
                 data: bytes,
                 timestamp: float):
        
        self._parent = parent
        self.data = data
        self.timestamp = timestamp
        
    @property
    def ndim(self):
        return self.data.ndim
        