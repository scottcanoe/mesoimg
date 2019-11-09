import os
from pathlib import Path
import time
import h5py
import numpy as np

from mesoimg.common import uint8, Clock, remove, Timer



def print_elapsed(T):
    print('Time elapsed: {:.2f} secs.'.format(T))

    
    
local_path = Path.home() / 'test.h5'
HD_root = Path('/media/pi/HD1')
path = HD_root / 'test.hdf5'
remove(path)

n_frames = 3000
n_xpix = 640
n_ypix = 480
n_channels = 1

if n_channels == 1:
    shape = (n_frames, n_ypix, n_xpix)
    template = np.empty([n_ypix, n_xpix], dtype=uint8)
else:
    shape = (n_frames, n_ypix, n_xpix, 3)
    template = np.empty([n_ypix, n_xpix, 3], dtype=uint8)

MB = np.prod(shape) * 1e-6
f = h5py.File(str(path), 'w')
print('Creating empty dataset with size {:.2f} MB...'.format(MB), flush=True)
timer = Timer(start=True, verbose=True)
dset = f.create_dataset('dset', shape, dtype=uint8)
timer.stop()

if n_channels == 1:
    template = np.empty([n_ypix, n_xpix], dtype=uint8)
else:
    template = np.empty([n_ypix, n_xpix, 3], dtype=uint8)

print('Starting transfer.')
timer.reset()
timer.start()
for i in range(n_frames):
    #frame = np.empty_like(template)
    frame = template
    dset[i] = frame
    timer.tic()

T = timer.stop()    
timestamps = timer.timestamps
rate = MB / T
IFIs = np.ediff1d(timestamps)
print("Transfer speed: {:.2f} MB/sec.".format(rate))
print('n_frames: {}'.format(n_frames))
print('secs: {:.2f}'.format(T))
print('FPS: {:.2f}'.format(n_frames / T))
print('median IFI: {:.2f} msec.'.format(1000 * np.median(IFIs)))
print('max IFI: {:.2f} msec.'.format(1000 * np.max(IFIs)))
print('min IFI: {:.2f} msec.'.format(1000 * np.min(IFIs)))

f.close()


import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
ax.plot(timestamps[:-1], IFIs)
plt.show()

















