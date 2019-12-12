import logging
import numpy as np
import os
import time
import numpy as numpy
from queue import Queue
import zmq
from mesoimg import *
from picamera import PiCamera

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('camera')
logger.level = logging.DEBUG


PAIR = 7010
FRAME_PUB = 7011
META_PUB = 7012

def close():
    for s in closing:
        s.close()
        if isinstance(s, Subscriber):
            time.sleep(0.1)
            s.join()


def put(data):
    out.put(data)


q = Queue(maxsize=100)
mq = Queue(maxsize=100)
out = Queue()

closing = []


cam = Camera(frame_q=q, meta_q=mq)
closing.append(cam)

ctx = zmq.Context()
pair = ctx.socket(zmq.PAIR)
pair.rcvtimeo = 5000
pair.connect(f'tcp://localhost:7010')
closing.append(pair)

# sub = ctx.socket(zmq.SUB)
# sub.connect(f'tcp://localhost:7011')
# sub.subscribe(b'')
# sub.rcvtimeo = 1000
# closing.append(sub)

sub = Subscriber(recv=recv_frame, callback=put, copy=False)
sub.connect(f'tcp://localhost:{FRAME_PUB}')
sub.subscribe(b'')
sub.start()
closing.append(sub)

fm = Frame.zeros([5, 5])
















