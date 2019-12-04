import io
import os
from pathlib import Path
import time
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from mesoimg import *


cam = None
resolution = (640, 480)
framerate = 30.0
channels = 'g'
duration = 2.0

h5_path = Path('/media/pi/HD1/mov.h5')
mpeg_path = Path('/media/pi/HD1/mov.mp4')
h264_path =  Path('/media/pi/HD1/mov.h264')
raw_path = Path('/media/pi/HD1/mov.raw')

for p in (h5_path, mpeg_path, raw_path, h264_path):
    if p.exists():
        p.unlink()
            
#cam = Camera()
#cam.reset()
#cam.channels = 'rgb'
cam = PiCamera()
cam.resolution = (486, 486)
cam.capture('/home/pi/test.jpg', 'rgb')

#cam._cam.start_recording(out, 'rgb')
#cam._cam.wait_recording(1)
#cam._cam.stop_recording()
#cam.capture(out, 'rgb')
#cam.start_recording(out, 'rgb')
#cam.wait_recording(1)
#cam.stop_recording()


#mov = read_raw(raw_path, (640, 480))
#imageio.mimwrite(mpeg_path, mov, fps=30)

#if success:
#    f = h5py.File(str(h5_path), 'r')
#    dset = f['data']
#    mov = dset[:]
#    n_frames = dset.attrs['n_frames'] 
#    ts = f['timestamps'][:]
#    ifis = np.ediff1d(ts)
#    mean_fps = 1 / np.mean(ifis)
#    print(f'Recorded {n_frames} frames at {mean_fps} FPS')
#    imageio.mimwrite(mpeg_path, mov, fps=mean_fps)
#    f.close()


#cam.close()






