"""
Script that was used to determine how to package raw data into arrays
with the correct dimensions, RGB order, etc.
"""
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import picamera
from picamera import PiCamera


n_xpix = 640
n_ypix = 480
n_channels = 3
raw_dtype = '>u1'


def clear_path(path):
    path = Path(path)
    if path.exists():
        path.unlink()
    return path

def get_cam():
    cam = PiCamera(resolution=(n_xpix, n_ypix), framerate=30)
    time.sleep(2)
    return cam

def record(secs=1, path='/home/pi/mov.dat'):                
    cam = get_cam()
    path = clear_path(path)
    cam.start_recording(str(path), 'rgb')
    cam.wait_recording(secs)
    cam.stop_recording()
    cam.close()

def capture(path='/home/pi/im.dat'):
    cam = get_cam()
    path = clear_path(path)
    cam.capture(str(path), 'rgb')
    cam.close()
    

def imshow(im):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(im)
    ax1.set_title('all')

    rgb_axes = []
    for i in range(3):
        ax = fig.add_subplot(1, 4, i + 2)
        ax.imshow(im[:, :, i], cmap='gray')
        rgb_axes.append(ax)
        
    
record(1)
arr = np.fromfile('/home/pi/mov.dat', dtype=raw_dtype)
n_bytes = len(arr)
bytes_per_frame = n_xpix * n_ypix * n_channels
n_frames = n_bytes / bytes_per_frame
assert np.isclose(n_frames - int(n_frames), 0)
n_frames = int(n_frames)

mov = np.zeros([n_frames, n_ypix, n_xpix, n_channels], dtype=raw_dtype)
for i in range(n_frames):
    frame_data = arr[i*bytes_per_frame : (i+1)*bytes_per_frame]
    frame = frame_data.reshape([n_ypix, n_xpix, n_channels])
    mov[i] = frame

imshow(mov[0])
