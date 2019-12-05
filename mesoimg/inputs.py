from threading import Thread
import time
from typing import Callable, Optional, Union, Sequence, NamedTuple
import numpy as np
import zmq
from mesoimg.common import *


HOST = 'pi-meso.local'



class FrameSubscriber(Thread):



    def __init__(self,
                 ctx,
                 topic: str = 'frame',
                 hwm: int = 10,
                 timeout: float = 1.0,
                 start: bool = True,
                 ):

        super().__init__()


        self.sock = ctx.socket(zmq.SUB)
        self.sock.subscribe = topic.encode()
        self.sock.hwm = hwm
        self.sock.connect(f'tcp://{HOST}:{Ports.FRAME_PUB}')
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self.timeout = timeout
        self._terminate = False
        if start:
            self.start()


    def run(self):

        # Alias
        sock = self.sock
        poller = self.poller
        timeout = self.timeout

        self._terminate = False
        while not self._terminate:
            ready = dict(poller.poll(timeout))
            if sock in ready and ready[sock] == zmq.POLLIN:
                frame = sub_frame(sock)
                print(f'Got frame: {frame.index}', flush=True)

        sock.close()
        poller.unregister(sock)
        time.sleep(0.01)


    def close(self):
        self._terminate = True


