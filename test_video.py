"""
Script used to figure out how to stream video and get timestamps for each frame.
"""

import io
import os
from pathlib import Path
import time
from typing import Optional, Tuple, Union
import warnings
import matplotlib.pyplot as plt
import numpy as np
from mesoimg.camera import *
from mesoimg.common import *





def record(secs=1, path='/home/pi/mov.dat', **kw):
    cam = Camera(**kw).cam
    path = clear_path(path)
    cam.start_recording(str(path), 'rgb')
    cam.wait_recording(secs)
    cam.stop_recording()
    cam.close()


def capture(path='/home/pi/im.dat', **kw):
    cam = Camera(**kw).cam
    path = clear_path(path)
    cam.capture(str(path), 'rgb')
    cam.close()
    
    
def imshow_rgb(im):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(im)
    ax1.set_title('all')
    ax1.set_aspect('equal')
    rgb_axes = []
    for i in range(3):
        ax = fig.add_subplot(1, 4, i+2)
        ax.imshow(im[:, :, i], cmap='gray')
        rgb_axes.append(ax)
        ax.set_aspect('equal')
        
        


def test_custom_loop(**kw):
    
    duration = 3
    path = '/home/pi/mov.dat'

    cam = Camera(**kw).cam

    T = ['NULL']
    t_start = time.time()
    cam.start_recording(str(clear_path(path)), 'rgb')
    while time.time() - t_start < duration:
        frame = cam.frame
        ts = frame.timestamp
        if ts != T[-1]:
            # New frame being drawn.
            T.append(ts)
        
    cam.stop_recording()
    cam.close()
    T = T[1:]
    T = [ts for ts in T if ts is not None]
    print('num timestamps: {}'.format(len(T)))
    mov = load_raw(path)
    print('mov frames: {}'.format(mov.shape[0]))
    print('FPS = {:.2f}'.format(len(T) / duration))


def getframe(stream):
    """
    Extract an RGB frame from a BytesIO stream and then reset it.
    """
    
    # Read the data from the stream.
    data = stream.getvalue()
    
    # Do some sanity checks.
    n_frames, remainder = np.divmod(len(data), bytes_per_frame)
    if n_frames == 0:
        raise IOError('no frames to read from buffer.')
    elif n_frames > 1:
        raise IOError('more than one frame in buffer.')
    if remainder != 0:
        raise IOError('buffer contains strange amount of data.')

    # Clear the stream.
    stream.truncate(0)
    stream.seek(0)

    # Reshape data into 3-d array with shape (n_ypix, n_xpix, 3).
    frame = np.frombuffer(data, dtype='>u1')
    frame = frame.reshape([n_ypix, n_xpix, 3])
    return frame




cam = Camera(resolution=(640, 480), frame_rate=30)

stream = io.BytesIO()
buff = stream.getbuffer()
    

#path = Path.home() / 'calimg-db' / 'etc' / 'mov.dat'
#with open(str(path), 'rb') as f:
    #data = f.read()

#arr = np.fromfile(str(path), dtype=uint8)

#n_bytes = len(arr)
#bytes_per_frame = n_xpix * n_ypix * 3
#n_frames = n_bytes / bytes_per_frame
#assert np.isclose(n_frames - int(n_frames), 0)
#n_frames = int(n_frames)

#mov = np.zeros([n_frames, n_ypix, n_xpix, 3], dtype=uint8)
#for i in range(n_frames):
    #frame_data = arr[i * bytes_per_frame : (i + 1) * bytes_per_frame]
    #frame = frame_data.reshape([n_ypix, n_xpix, 3])
    #mov[i] = frame
    
    
    
#duration = 2
#stream = io.BytesIO()
#b = stream.getbuffer()
#frames = []
#timestamps = ['NULL']

#cam = get_cam()
#t_start = time.time()
#for foo in cam.capture_continuous(stream, 'rgb', use_video_port=True):

    #cur_t = time.time()
    #if cur_t - t_start >= duration:
        #break
    
    #frame = getframe(stream)


#timestamps = timestamps[1:]
#timestamps = [ts for ts in timestamps if ts is not None]
#n_timestamps = len(timestamps)
#print('num timestamps: {}'.format(n_timestamps))
#mov = load_mov(path)
#n_frames = mov.shape[0]
#print('mov frames: {}'.format(n_frames))
#print('FPS = {:.2f}'.format(n_frames / duration))

duration = 2
n_frames = None

cam = Camera()
frames = []
for fm in cam.iterframes(duration=duration, n_frames=n_frames):
    frames.append(fm)


