from pathlib import Path
from threading import Event, Lock, RLock, Thread
import time
from typing import Sequence, Union
import h5py
import numpy as np
from mesoimg import *


class H5Writer(Thread):
    """


    """

    def __init__(self,
                 path: PathLike,
                 shape: Sequence[int],
                 dtype: Union[str, type] = np.uint8,
                 ):
        super().__init__()

        self.lock = Lock()
        self.terminate = Event()

        self.path = Path(path)
        self.shape = shape
        self.dtype = dtype



    def start(self):

        # Alias
        path = Path(self.path)
        shape = self.shape
        dtype = self.dtype

        # Open file for writing
        self.file = h5py.File(str(path), 'w')

        self.max_frames = shape[0]
        self.dset = self.file.create_dataset('data', shape, dtype=dtype)
        self.dset.attrs['index'] = 0
        self.ts = self.file.create_dataset('timestamps', (shape[0],), dtype=float)

        self._n_received = 0
        self._complete = False

        print(f'Starting H5Writer at {path}')
        super().start()



    def run(self) -> None:

        while not self.terminate.is_set():
            pass
        self.file.close()
        print(f'Finished writing to file {self.path}')


    def stop(self) -> None:
        self.terminate.set()
        time.sleep(0.1)


    def close(self) -> None:
        self.stop()


    def __call__(self, frame: Frame) -> None:


        if self._complete:
            self.terminate.set()
            return

        with self.lock:

            index = self.dset.attrs['index']
            self.dset[index, ...] = frame[:]
            self.ts[index] = frame.timestamp
            self.dset.attrs['index'] += 1
            self._n_received += 1
            if self._n_received % 30 == 0:
                print(f'Wrote frame: {self._n_received}')

        if self.dset.attrs['index'] >= self.max_frames:
            self._complete = True
            self.terminate.set()


path = Path.home() / 'test3.h5'
shape = (5*600, 1232, 1640)
f = H5Writer(path, shape, dtype=np.uint8)
f.start()

sub = Subscriber(recv_frame)
sub.connect(f'tcp://pi-meso.local:{Ports.CAM_FRAME}')
sub.subscribe(b'')
sub.callback = f
sub.start()

#path = Path.home() / 'test.h5'
#f = h5py.File(str(path), 'r')
#dset = f['data']
#mov = dset[:]
#f.close()

#n_frames, ypix, xpix = mov.shape
##xmid = int(xpix/2)
##ymid = int(ypix/2)
#xmid = 1000
#ymid = 616
#q = 486
#mov = mov[:, ymid-q:ymid+q, xmid-q:xmid+q]
#ptile = np.percentile(mov, 99.9)
#mov = mov / ptile
#mov[mov > 255] = 255
#p = Path.home() / 'test.mp4'
#write_mp4(p, mov, fps=15)

