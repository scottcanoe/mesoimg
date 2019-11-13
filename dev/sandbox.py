import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mesoimg import *
import h5py


secs = 2
SSD = True
raw = True
FPS = 40
resolution = (640, 480)


#path = Path('/media/pi/HD1/') if SSD else Path('/home/pi/')
#path = path / 'mov.raw' if raw else path / 'mov.h264'
#format = 'rgb' if raw else 'h264'
path = Path('/media/pi/HD1/mov.h5')
if path.exists():
    path.unlink()

cam = Camera()

out = FrameBuffer(cam)
maxframes = int(secs * FPS * 1.1)
out.connect(path, maxframes)

cam.start_recording(out, 'rgb')
cam.wait_recording(secs)
cam.stop_recording()
out.close()
cam.close()


ts = out.ts
ts = out.out._timestamps
n_frames = len(ts)
print(f'{n_frames/secs}')

f = h5py.File(path, 'r')
dset = f['data']
im = dset[20]
f.close()
plt.ion()
plt.imshow(im)

#if raw:
#    mov = read_raw(path, resolution)
#else:
#    mov = read_h264(path)

#n_frames = mov.shape[0]
#print(f'{n_frames/secs}')

