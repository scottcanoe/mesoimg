import io
import os
from pathlib import Path
from typing import Optional, Union, Sequence
import matplotlib.pyplot as plt
import numpy as np
from mesoimg.common import *
import time

import h5py





class H5WriteStream:
    
    
    def __init__(self,
                 url: URL,
                 shape: Sequence[int],
                 mode: str = 'a',                 
                 dtype: Optional[Union[str, np.dtype]] = np.uint8,
                 ):

        url = urlparse(url)
        path = url.path
        dpath = url.fragment if url.fragment else 'data'
        dpath = dpath[1:] if dpath.startswith('/') else dpath

        self.url = url
        self.path = path
        self.dpath = dpath
        self.file = h5py.File(path, mode)
        if dpath in self.file.keys():
            del self.file[dpath]
            
        # Try to create the dataset.
        self.shape = shape
        self.dtype = dtype
        
        self.file.create_dataset(dpath, shape, dtype=dtype)
        self.dset = self.file[dpath]      
        self._index = 0
        self._timestamps = []

                        
    def write(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """
        The client calls this to dump data.
        """
        
        self.dset[self._index] = frame
        self._index += 1
        self._timestamps.append(timestamp)
        return 1
        
        
    def flush(self):
        pass
        
        
    def close(self):
        self.file.close()

        
#        

#class Socket:

#    
#    def __init__(self, host: Union[URL, Host]):

#        if isinstance(host, Host):
#            self.host = host
#        else:
#            url = urlparse(host)
#            path = url.path
#                        
#        
#    
#    def recv(self):
#        """
#        The owning client listens for a response 
#        with recv.
#        """
#        
#        
#    def send(self, data):
#        """
#        The owning client uses this to send to the 
#        resource. Blocking?
#        """
#        pass


#    def connect(self, *args, **kw):
#        """
#        When the client calls this, open the resource
#        and prepare it for access.
#        """
#        pass


#path = '/media/pi/HD1/mov.hdf5'
#cam = Camera()
#out = OutputStream(cam)














