from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import imageio
from mesoimg.common import PathLike, fspath



class ImageViewer:


    def __init__(self, cam: 'Camera', width: float = 8):

        plt.ion()

        n_xpix, n_ypix = cam.resolution
        height = width * (n_ypix / n_xpix)        
        
        self.fig = plt.figure(figsize=(width, height))
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.im = self.ax.imshow(np.zeros([n_ypix, n_xpix, 3], dtype='u1'))
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        plt.pause(0.1)
    
    @property
    def closed(self):        
        return not plt.fignum_exists(self.fig.number)
    
    def update(self, data: np.ndarray, wait: float = 0.05) -> None:        
        self.im.set_data(data)
        plt.pause(wait)
        
    def save(self, path: PathLike) -> None:

        path = str(path) if isinstance(path, Path) else path
        data = np.array(self.im.get_array())
                
        
    

