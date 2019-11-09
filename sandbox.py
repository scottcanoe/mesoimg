import io
import logging
import time
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import picamera
from mesoimg import MesoCam
from mesoimg.common import *


logging.basicConfig(level=logging.INFO)

"""
Interactive backends may be:
   Matplotlib: 'matplotlib', 'mpl', 'pyplot'
   OpenCV: 'opencv', 'cv', 'cv2'
Non-interactive backends will use matplotlib and save a figure.   
"""

settings = {}
settings['imshow.backend'] = 'matplotlib'
settings['preview.backend'] = 'opencv'

NORMALIZE_BACKEND_NAMES = {}
for alias in ('matplotlib', 'mpl', 'pyplot'):
    NORMALIZE_BACKEND_NAMES[alias] = 'matplotlib'

for alias in ('opencv', 'cv', 'cv2'):
    NORMALIZE_BACKEND_NAMES[alias] = 'opencv'
    


def imshow(im, *, resolution=(640, 480), title='', **kw):

    if isinstance(im, bytes):
        im = np.frombuffer(im, uint8)

    if im.ndim == 1:
        n_xpix, n_ypix = resolution
        bytes_per_frame = n_xpix * n_ypix
        if len(im) == bytes_per_frame:
            im = im.reshape([n_ypix, n_xpix])
        else:
            im = im.reshape([n_ypix, n_xpix, 3])
        

    plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(im, **kw)
    ax.set_title(title)
    ax.set_aspect('equal')
    return fig, ax


def imshow_opencv(im, title='image', **kw):
    import cv2
    im = im[..., [2, 1, 0]].copy()
    cv2.imshow(title, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def reshape(im, resolution=(640, 480)):

    n_xpix, n_ypix = resolution
    
    mono_bytes = n_xpix * n_ypix
    rgb_bytes = mono_bytes * 3

    if len(im) == mono_bytes:
        return im.reshape([n_ypix, n_xpix])
    elif len(im) == rgb_bytes:
        return im.reshape([n_ypix, n_xpix, 3])    
    else:
        raise ValueError
   
    
    


def record(cam=None, *, max_frames=None, max_secs=None, verbose=False):

    # Handle max_frames and max_secs arguments.
    max_frames = np.inf if max_frames is None else max_frames
    max_secs = np.inf if max_secs is None else max_secs
    if max_frames == np.inf and max_secs == np.inf:
        raise ValueError('max_frames and max_secs cannot both be infinite.')

    # Setup state and record variables.
    frame_counter = 0
    frames = []
    clock = Clock()
    timestamps = []
    stream = io.BytesIO()

    if cam is None:
        cam_created = True
        cam = MesoCam()
    else:
        cam_created = False

    logging.info('Beginning continuous capture.')
    analog_gains = []
    awb_gains = []
    exposure_speeds = []
    try:
        clock.start()
        for foo in cam.capture_continuous(stream, 'rgb', use_video_port=True):

            frame_counter += 1
            ts = clock.time()

            if frame_counter > max_frames or ts > max_secs:
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



cam = MesoCam()

d = record(cam, max_frames=60, verbose=True)
print(repr(cam))

timestamps = d['timestamps']
frames = d['frames']

analog = np.array([float(val) for val in d['analog_gains']])
awb_gains = d['awb_gains']
red = np.array([float(tup[0]) for tup in awb_gains])
blue = np.array([float(tup[1]) for tup in awb_gains])

fig, ax = plt.subplots()
ax.plot(analog, c='black', lw=2)
ax.plot(red, c='red')
ax.plot(blue, c='blue')
speeds = d['exposure_speeds']

cam.close()






