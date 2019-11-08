import io
import logging
import time
import numpy as np
import picamera
from mesoimg.common import *

logging.basicConfig(level=logging.INFO)


class MesoCam(picamera.PiCamera):

    """
    
    Make a TCP server?
    
    
    """

    def __init__(self, resolution=(640, 480),
                       framerate=30.0,
                       channels='g',
                       sleep=1,
                       **kw):
        super().__init__(resolution=resolution, framerate=framerate, **kw)

        logging.info('Initializing MesoCam.')

        # Handle channel info.
        if not all(c in 'rgb' for c in channels):
            raise ValueError("Invalid channel argument '{}'.".format(channels))
        self._default_channels = channels
        
        # Let camera warm up.
        time.sleep(sleep)
        
        
    def _prepare(self, format=None, channels=None):

        # Handle channels.
        channels = self._default_channels if channels is None else channels
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
        
         
cam = MesoCam()
n_xpix, n_ypix = cam.resolution

stream = io.BytesIO()


max_frames = None
max_secs = 5

max_frames = np.inf if max_frames is None else max_frames
frame_count, frames = 0, []

max_secs = np.inf if max_secs is None else max_secs
clock, T = Clock(), []

logging.info('Beginning continuous capture.')

clock.start()
for foo in cam.capture_continuous(stream, 'rgb', use_video_port=True):
 
    frame_count += 1
    cur_t = clock.time()
    
    if frame_count > max_frames or cur_t > max_secs:
        break
        
    # Grab the buffered data, and remove non-green channels.
    arr = stream.getvalue()
    stream.seek(0)

    # Keep only green channel.
    arr = arr[1::3]    

    # Update frames and timestamps.
    frames.append(arr)
    T.append(cur_t)


t_stop = clock.stop()
cam.close()

T = np.array(T)
dt = np.ediff1d(T)

print('n_frames: {}'.format(frame_count))
print('secs: {}'.format(t_stop))
print('FPS: {}'.format(frame_count / t_stop))

#frames = [np.frombuffer(fm, dtype=uint8) for fm in frames]
#f = frames[0]
#im = f[1::3]

import matplotlib.pyplot as plt
plt.ion()

im = frames[0]
im = np.frombuffer(im, dtype=uint8)
plt.imshow(im)
plt.show()






















