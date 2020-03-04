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
        """
        Callback called by frame subscriber socket thread.

        """

        if self._complete:
            self.terminate.set()
            return

        with self.lock:

            index = self.dset.attrs['index']
            self.dset[index, ...] = frame.data
            self.ts[index] = frame.time
            self.dset.attrs['index'] += 1
            self._n_received += 1
            if self._n_received % 30 == 0:
                print(f'Wrote frame: {self._n_received}')

        if self.dset.attrs['index'] >= self.max_frames:
            self._complete = True
            self.terminate.set()



path = Path.home() / 'mov-03-04-post-impact-3.h5'
secs = 5 * 60
fps = 15

nframes = int(secs * fps)
shape = (nframes, 1232, 1640)
writer = H5Writer(path, shape, dtype=np.uint8)
writer.start()

sub = Subscriber(recv_frame)
sub.connect(f'tcp://pi-meso.local:{Ports.CAM_FRAME}')
sub.subscribe(b'')
sub.callback = writer
sub.start()

