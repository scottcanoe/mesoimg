#import matplotlib
#matplotlib.use('pdf')
#import matplotlib.pyplot as plt
import numpy as np
from mesoimg.camera import *
from mesoimg.common import *

#cam = Camera(resolution=(640, 480), sensor_mode=7, frame_rate=30)
#path = str(clear_path('/home/pi/im.dat'))
#im = load_raw(path)
#fig, ax = plt.subplots()
#ax.imshow(im[:, :, 0])
#fig.savefig('/home/pi/fig.png')

from picamera import PiCamera
#cam = PiCamera(resolution=(640, 480), framerate=30, sensor_mode=7)

cam = Camera(resolution=(640, 480), sensor_mode=0, frame_rate=30)
#duration = None
#n_frames = 60
#clear_path('/home/pi/mov.h264')

#frames = []
#clock = Clock(start=True)
#for fm in cam.iterframes(duration=duration, n_frames=n_frames):
   ##print(clock.time())
   #frames.append(fm)

#t_stop = clock.stop()
#n_frames = len(frames)

#print('frames: {}'.format(n_frames))
#print('secs: {}'.format(t_stop))
#print('fps: {}'.format(n_frames / t_stop))

#cam.close()





