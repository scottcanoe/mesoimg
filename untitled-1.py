import io
import os
from pathlib import Path
import socket
import time
from typing import (Any,
                    Callable,
                    Iterable,
                    List,
                    Mapping,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)
import numpy as np
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
#from mesoimg.camera import Camera

#path = remove('/home/pi/foo.jpeg')


#cam = Camera()
#cam.capture(path, show=True)
#cam.close()                              
from typing import NamedTuple

class A(NamedTuple):
    x: int = 0
    y: int = 1
    
a = A()    
