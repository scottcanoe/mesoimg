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
from mesoimg.common import (ArrayTransform,
                            PathLike,
                            uint8,
                            pathlike,
                            remove,
                            read_json,
                            write_json,
                            squeeze,
                            Clock)
from mesoimg.camera import Camera

path = remove('/home/pi/foo.jpeg')

#cam = PiCamera(resolution=(640, 480),
#               framerate=30.0,
#               sensor_mode=0)

cam = Camera()
cam.capture(path, show=True)


#cam.close()                              
