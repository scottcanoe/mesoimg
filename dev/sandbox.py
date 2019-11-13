import io
import os
from pathlib import Path
import PIL
import matplotlib.pyplot as plt
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from mesoimg import *
import time

plt.ion()



def draw(mat):
    global fig, ax, im, frame
    if fig is None:
        fig, ax = plt.subplots()
    
    images = ax.get_images()
    if len(images) == 0:
        im = ax.imshow(mat)
        ax.set_aspect('equal')
        return
    assert len(images) == 1    
    cur_shape = images[0].get_array().shape
    if mat.shape == cur_shape:
        im.set_data(frame)
    else:
        ax.clear()
        im = ax.imshow(mat)
    ax.set_aspect('equal')
     

def snap() -> np.ndarray:

    global frame
    try:
        buf = PiRGBArray(cam)
        cam.capture(buf, 'rgb', use_video_port=True)        
        frame = buf.array
        draw(frame)
    except:
        cam.close()
        raise


def record(n_frames = 30):

    global frame
    #buf = PiRGBArray(cam)
    buf = io.BytesIO()
    try:
        frame_counter = 0
        for foo in cam.capture_continuous(buf,
                                          'rgb',
                                          use_video_port=True):
            #frame = buf.array
            frame = buf.getvalue()
            buf.seek(0)
            buf.truncate(0)
            frame_counter += 1
            if frame_counter >= n_frames:
                break
            buf.seek(0)
            buf.truncate(0)
    except:
        cam.close()
        raise
    #draw(frame)
    return frame



from mesoimg.camera import ImageBuffer

cam = Camera()
buf = ImageBuffer(cam)
im = cam.preview()
#cam.close()



