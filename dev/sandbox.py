import io
import os
from pathlib import Path
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from picamera import PiCamera
from mesoimg import *
from mesoimg.outputs import *
from mesoimg.common import *
from mesoimg.errors import *            


cam = None
resolution = (640, 480)
framerate = 30.0
channels = 'g'
duration = 5.0

h5_path = Path('/media/pi/HD1/mov.h5')
mpeg_path = Path('/media/pi/HD1/mov.mp4')

for p in (h5_path, mpeg_path):    
    if p.exists():
        p.unlink()
            
try:

    cam = Camera(resolution=resolution,
                 framerate=framerate,
                 channels=channels)
#    cam.exposure_mode = 'off'
#    cam.awb_mode = 'off'
#    cam.shutter_speed = 1000
    cam.record(h5_path, duration, overwrite=True)
    success = True

except Exception:

    cam.close() if cam else None
    success = False
    raise
    
if success:
    f = h5py.File(str(h5_path), 'r')
    dset = f['data']
    mov = dset[:]
    n_frames = dset.attrs['n_frames'] 
    ts = f['timestamps'][:]
    ifis = np.ediff1d(ts)
    mean_fps = 1 / np.mean(ifis)
    print(f'Recorded {n_frames} frames at {mean_fps} FPS')
    imageio.mimwrite(mpeg_path, mov, fps=mean_fps)
    f.close()


#cam.close()






