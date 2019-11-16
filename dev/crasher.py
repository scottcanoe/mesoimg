import numpy as np
import zmq
from mesoimg import *


cam = Camera()
out = FrameBuffer(cam)
out._in_shape = (600, 600, 3)
cam.start_recording(out)
cam.wait_recording(1)
cam.stop_recording()
