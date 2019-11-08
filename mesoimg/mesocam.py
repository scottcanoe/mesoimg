import io
import logging
import time
import numpy as np
import picamera
from mesoimg.common import *

logging.basicConfig(level=logging.INFO)


class MesoCam(picamera.PiCamera):

    """
    Camera that s
    Make a TCP server?


    """

    default_channels = 'g'

    def __init__(self, resolution=(640, 480),
                       framerate=30.0,
                       sensor_mode=7,
                       warm_up=True,
                       warm_up_secs=2,
                       **kw):

        logging.info('Initializing MesoCam.')

        super().__init__(resolution=resolution,
                         framerate=framerate,
                         sensor_mode=sensor_mode,
                         **kw)

        # Let camera warm up.
        if warm_up:
            time.sleep(warm_up_secs)


    def prepare(self):
        raise NotImplementedError


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
        s  = '       MesoCam      \n'
        s += '--------------------\n'
        s += 'resolution: {}\n'.format(self.resolution)
        s += 'framerate: {}\n'.format(self.framerate)
        s += 'sensor_mode: {}\n'.format(self.sensor_mode)
        s += 'closed: {}\n'.format(self.closed)

        return s


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








