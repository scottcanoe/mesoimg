from pathlib import Path
import time
import numpy as np
from picamera import PiCamera
from mesoimg.common import *


def opencam(sensor_mode=7, resolution=(480, 480), **kw):
	cam = PiCamera(sensor_mode=sensor_mode,
		           resolution=resolution,
		           framerate=30,
		           **kw)
	cam.video_denoise = False
	time.sleep(2)
	return cam

def convert(out, resolution):
	mov = read_raw(out, resolution)	
	n_frames = mov.shape[0]
	fps = n_frames / secs
	print(f'fps: {fps}')
	write_mp4(out.parent / 'mov.mp4', mov, fps=fps)


resolution = (480, 480)
cam = opencam(resolution=resolution)


out = Path('/home/pi/out.raw')
secs = 2

try:
	cam.start_recording(str(out), 'rgb')
	cam.wait_recording(secs)
	cam.stop_recording()
except:
	cam.close()

# if not cam.closed:
# 	cam.close()

convert(out, resolution)
