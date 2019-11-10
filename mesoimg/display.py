from typing import (Any,
                    Callable,
                    Iterable,
                    List,
                    Mapping,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)

import imageio
import matplotlib.pyplot as plt
import numpy as np
from picamera.array import PiRGBArray
from mesoimg.common import PathLike, pathlike


class ImageViewer:


    def __init__(self,
                 data: Union[PathLike, np.ndarray, PiRGBArray],
                 cmap: Optional[str] = None):

        if isinstance(data, PiRGBArray):
            data = data.array
        elif isinstance(data, np.ndarray):
            pass
        elif pathlike(data):
            data = imageio.imread(data)
        else:
            raise NotImplementedError

        assert data.ndim <= 3
        n_ypix, n_xpix = data.shape[0], data.shape[1]
        aspect = n_xpix / n_ypix
        width = 8
        height = width / aspect        
        
        # Initialize viewing area.
        plt.ion()
        plt.tight_layout()
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(1, 1, 1)        
        im = ax.imshow(data, cmap=cmap)
        ax.set_aspect('equal')
        
        self.data = data
        self.fig = fig
        self.ax = ax
        self.im = im
        self.cmap = cmap

        
        
    def close(self):
        
        fig = self.fig
        self.fig = None
        self.ax = None
        self.im = None
        plt.close(fig)
        
                    
        
