from collections import UserDict
import io
import logging
import os
from pathlib import Path
import socket
import time
from typing import (Any,
                    Callable,
                    Iterable,
                    List,
                    Mapping,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)
import weakref
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
                            squeeze,
                            Clock)





class Property:
    
    def __init__(self, key):
        self.key = key
    
    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        return obj[self.key]
    
    def __set__(self, obj, val):
        obj[self.key] = val
    
    def __del__(self, obj):
        del obj[self.key]



class Config(UserDict):
    
        
    #: Width and height of imagery in pixels.
    resolution = Property('resolution')
    
    #: Limiting framerate.
    framerate = Property('framerate')
    
    #: Sensor mode indicates things like resolution, binning, etc.
    sensor_mode = Property('sensor_mode')
    
    #: Whether to use RGB data rather than compressed representations.
    raw = Property('raw')
    
    #: If in raw mode, define which channels to be discarded (if any).
    channels = Property('channels')
    
    #: Analog gain controls.
    exposure_mode = Property('exposure_mode')
    
    #: Last frame readout time (actually line readout time * num. lines). 
    shutter_speed = Property('shutter_speed')
    
    #: White-balance mode.
    awb_mode = Property('awb_mode')
    
    #: Two-tuple of white-balance gains (red, blue).
    awb_gains = Property('awb_gains')
    
    #: Whether to horizontally flip image (in GPU).
    hflip = Property('hflip')
    
    #: Whether to vertically flip image (in GPU).
    vflip = Property('vflip')
               
    #: Whether to use the video port (faster).
    use_video_port = Property('use_video_port')
    
           
    def __init__(self,
                 config: Union[PathLike, Mapping] = 'default',
                 **kw):
        
        # Initialize dict with defaults.
        self.data = CAM_CONFIGS['default'].copy()
                
        # Update from a supplied dict/Config object.
        if isinstance(config, (dict, Config)):
            self.data.update(config)
                
        # Update from a hard-coded setting.
        elif config in CAM_CONFIGS.keys():
            self.data.update(CAM_CONFIGS[config])

        # Update from settings on disk.
        else:
            aux = read_json(config)
            self.data.update(aux)
        
        # Finally, let any supplied keyword args override other values.
        self.data.update(kw)

        
    def save(self, path: PathLike, **kw) -> None:
        write_json(path, self.data, **kw)





# Define default config.
#----------------------------------

CAM_CONFIGS = {}

CAM_CONFIGS['default'] = {
    'resolution'     : None,
    'framerate'      : None,
    'sensor_mode'    : 0,
    'raw'            : False,
    'channels'       : 'rgb',
    'exposure_mode'  : 'auto',
    'shutter_speed'  : 0,
    'awb_mode'       : 'auto',
    'awb_gains'      : (0, 0),
    'hflip'          : False,
    'vflip'          : False,
    'use_video_port' : True,
}


# Define still image formats.
#----------------------------------
CAM_CONFIGS['still'] = {
    'use_video_port' : False,
}


# Define raw image/video formats.
#----------------------------------

CAM_CONFIGS['rgb'] = {
    'resolution'     : (640, 480),
    'framerate'      : 30.0,
    'sensor_mode'    : 7,
    'raw'            : True,
    'channels'       : 'rgb',
    'exposure_mode'  : 'fix',
    'awb_mode'       : 'fix',
    'use_video_port' : True,
}

for channel in ('r', 'g', 'b'):
    d = CAM_CONFIGS['rgb'].copy()
    d['channels'] = channel
    CAM_CONFIGS[channel] = d










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

    
    
    
    def __init__(self,
                 config: Union[PathLike, Mapping] = 'default',
                 **kw):
        
        logging.info('Initializing MesoCam.')
        self._config = Config(config, **kw)
        super().__init__(resolution=self.config.resolution,
                         framerate=self.config.framerate,
                         sensor_mode=self.config.sensor_mode)


    @property
    def config(self) -> Config:
        return self._config
    
    
    @config.setter
    def config(self, c: Union[PathLike, Mapping]) -> None:
        self._config = Config(config)
        
        
    def update_config(self, attrs: Iterable[str]) -> None:
        for key in attrs:
            self.config[key] = getattr(self, key)
            
            
    def prepare_to_capture(self):
        self._ready_to_capture = True


    def prepare_to_record(self):
        self._ready_to_record = True


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
        clock = Clock()
        signals = [] if signals is None else signals
        df = {sig: np.zeros(max_frames) for sig in signals}
        
        logging.info('Beginning recording.')
        try:
            clock.start()
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
             
                    
    def capture(self,
                out: Union[None, PathLike] = None,
                show: bool = True,
                **kw):


        config = self.config.copy()
        if kw:
            config.update(kw)
        use_video_port = config.use_video_port
        
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
        

    def _get_format(self, out: Any, config: Config) -> str:
        pass

        
    def _get_out_shape(self, out: Any, config: Config) -> Sequence[int]:

        n_xpix, n_ypix = self.resolution
        if not config.raw or config.channels == 'rgb':
            return [n_ypix, n_xpix, 3]
        else:
            return [n_ypix, n_xpix]

    
    
    def _get_channel_extract(self,
                                config: Config) -> Optional[ArrayTransform]:
        
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
            return 'MesoCam (closed)'            

        s  = '       MesoCam      \n'
        s += '--------------------\n'
        
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



class ImageViewer:


    def __init__(self,
                 data: Union[PathLike, np.ndarray, PiRGBArray],
                 figsize: Tuple[float, float] = (8, 8),
                 cmap: Optional[str] = None):

        if isinstance(data, PiRGBArray):
            data = data.array
        elif pathlike(data):
            data = imageio.imread(data)
        else:
            raise NotImplementedError

        # Initialize viewing area.
        plt.ion()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)        
        im = ax.imshow(data, cmap=cmap)
        ax.set_aspect('equal')
        
        self.data = data
        self.fig = fig
        self.ax = ax
        self.im = im
        self.cmap = cmap
        plt.show()
        
        
    def close(self):
        
        fig = self.fig
        self.fig = None
        self.ax = None
        self.im = None
        plt.close(fig)
        
                    
        

        
    
        
    
        





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












