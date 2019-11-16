from pathlib import Path
from threading import Thread
import time
import matplotlib.pyplot as plt
import numpy as np
import imageio
from mesoimg.common import PathLike
from mesoimg.outputs import Frame




class Preview:


    def __init__(self,
                 cam: 'Camera',
                 frame_buffer: 'FrameBuffer',
                 width: float = 8,   # approx. figure width ("inches")
                 cmap='inferno'):    # colormap to use if not not RGB.

        self.cam = cam
        self.frame_buffer = frame_buffer

        self.frame_shape = frame_buffer.out_shape
        n_ypix, n_xpix = self.frame_shape[1:3]
        height = width * (n_ypix / n_xpix)        

        self.n_channels = len(frame_buffer.channels)
                                
        if self.n_channels == 1:
            raise NotImplementedError

        
        plt.ion()
        
        self.fig = plt.figure(figsize=(width, height))
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        self.fig_number = self.fig.number

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        #blank = np.zeros(self.frame_shape, dtype=np.uint8)
        #self.im = self.ax.imshow(blank)

        plt.pause(1.0)
        self.start()
    
    
    @property
    def closed(self):        
        return not plt.fignum_exists(self.fig_number)
    
    
    def run(self):

        while not self.closed:
            
            # Acquire a frame.
            with self.frame_buffer.lock:
                frame = self.frame_buffer.frame
            if frame is None:
                time.sleep(1)
                continue

            # Update the plot.
            #self.im.set_data(frame.data)
            #plt.pause(0.1)
                

    def save(self, path: PathLike) -> None:

        path = str(path) if isinstance(path, Path) else path
        data = np.array(self.im.get_array())
        imageio.imwrite(path, data)
        
                
        
    

