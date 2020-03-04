from collections import deque
import queue
from queue import Queue
import threading
from threading import Condition, Event, Lock, RLock, Thread
import time
import numpy as np
import matplotlib.pyplot as plt
import zmq
from mesoimg.common import *
from mesoimg.messaging import *




def close():

    r.stop()


def callback(data: np.ndarray) -> None:
    """
    Doctring for f1
    """
    print(f'callback 1: {data}')
    q.put(data)


FRAME_PUB = 7011
q = queue.Queue()

r = Subscriber(recv_frame)
r.connect(f'tcp://pi-meso.local:{FRAME_PUB}')
r.subscribe(b'')
r.callback = callback
r.start()


