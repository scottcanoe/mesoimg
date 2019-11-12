import os
from pathlib import Path
import numpy as np
from mesoimg.timing import time, sleep, Timer

dtype = np.dtype('u1')

path = Path.home() / 'test.raw'
n_ypix, n_xpix = 640, 480
n_frames = 1000

bytes_per_frame = n_xpix * n_ypix
bufsize = -1


mov = np.empty([n_frames, n_ypix, n_xpix], dtype=dtype)
#arr = mov.tobytes()

f = open(path, 'wb')
try:
    
    tm = Timer('tm', verbose=True).start()
    for i in range(n_frames):
        f.write(mov[i])
        tm.tic()
    tm.stop()
except:
    f.close()
    raise

f.close()
