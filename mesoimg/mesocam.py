import io
import logging
import time
import numpy as np
import picamera
from mesoimg.common import *

logging.basicConfig(level=logging.INFO)

settings = {}

settings['mesocam.server.address'] = ['192.168.0.9', '65111']


class MesoCam(picamera.PiCamera):

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
    # Default is to use only green channel.
    default_channels = 'g'

    def __init__(self, resolution=(640, 480),
                       framerate=30.0,
                       sensor_mode=7,
                       **kw):

        logging.info('Initializing MesoCam.')

        super().__init__(resolution=resolution,
                         framerate=framerate,
                         sensor_mode=sensor_mode,
                         **kw)

        self._ready_to_capture = False
        self._ready_to_record = False

        # Disable white balance.
        #self.awb_gains = 'off'


    def start_server(self, port=

    def prepare_to_capture(self):
        self._ready_to_capture = True

    def prepare_to_record(self):

        self._ready_to_record = True


    #def fix_white_balance
    def snapshot(self, out=None, format='rgb', preview=False, **kw):

        if not self._ready_to_capture:
            self.prepare_to_capture()

        if out is None:
            out = io.BytesIO()

        picamera.PiCamera.capture(self, out, format, **kw)
        if preview:
            raise NotImplementedError
            
        n_xpix, n_ypix = self.resolution
        im = np.frombuffer(out.getvalue(), dtype=uint8).reshape([n_ypix, n_xpix, 3])
        return im
        
         
            
            



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


    def _prepare_encoder(self, format=None, channels=None):

        # Handle channels.
        channels = self.default_channels if channels is None else channels
        if len(channels) < 1 or not all(c in 'rgb' for c in channels):
            raise ValueError("Invalid channel argument '{}'.".format(channels))
        n_channels = len(channels)
        channel_indices = np.array(['rgb'.find(c) for c in channels])

        # Handle frame shape.
        n_xpix, n_ypix = self.resolution
        bytes_per_frame = n_xpix * n_ypix * n_channels
        if n_channels == 1:
            frame_shape = (n_ypix, n_xpix)
        else:
            frame_shape = (n_ypix, n_xpix, n_channels)



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



class BufferTransform:

    def __init__(self):
        pass


class ChannelExtractor(BufferTransform):

    """
    Convert a flat RGB frame into a flat, C-contiguous ndarray.

    """


    def __init__(self, channel):

        if channel in ('red', 'green', 'blue'):
            channel = channel[0]

        if isinstance(channel, str):
            if channel not in ('r', 'g', 'b'):
                raise ValueError("invalid channel:'{}'.".format(channel))
            self._channel = channel
            self._channel_index = 'rgb'.find(channel)
        elif isinstance(channel, int):
            if channel not in (0, 1, 2):
                raise ValueError("invalid channel: '{}'.".format(channel))
            self._channel = 'rgb'[channel]
            self._channel_index = channel
        else:
            msg  = "channel argument must be one or 'r', 'g', 'b' or 0, 1, 2. "
            msg += "you gave {}.".format(channel)
            raise ValueError(msg)

    @property
    def channel(self):
        return self._channel

    @property
    def channel_index(self):
        return self._channel_index

    def __init__(self, arr):
        mem = arr[self._channel_index::3]

        return mem

























