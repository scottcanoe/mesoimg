from queue import Queue
from threading import Event, Lock, Thread
from time import perf_counter as clock
import numpy as np
import zmq
from mesoimg import *




class MesoServer:

    COMMAND_PORT = 7000
    DATA_PORT = 7001


    def __init__(self, connect=True):
        self.cam = None
        self.open_sockets()
        self.run()
            
    
    def run(self):

        while True:
            self.cmd_sock.recv_string()
                    
    
    
def send_frame(socket,
               data: np.ndarray,
               index: int,
               timestamp: float,
               flags: int = 0,
               copy: bool = True,
               track: bool = False,
               ) -> None:

    md = {'shape': data.shape,
          'dtype': str(data.dtype),
          'index': index,
          'timestamp' : timestamp}
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(data, flags, copy=copy, track=track)    
    
        
                
def run_server(event):
    """
    Run in a thread, probably.
    """
    while not event.is_set():
        pass
    
import time

class Looper(Thread):
    
    def __init__(self):
        super().__init__()
        self.event = Event()
        self.lock = Lock()
        self.count = 0
        self.terminated = False
        self.start()
                
    def run(self):
        
        while not self.event.is_set():
            with self.lock:
                self.count += 1
            time.sleep(1)
            

"""
Camera will have the events, outside looper will have the sockets.
While the camera is busy recording, the outside loop can query
for input.

"""    

PORT = 7008
context = zmq.Context()
sock = context.socket(zmq.REQ)
sock.bind(f'tcp://*:{PORT}')

#loop = Looper()
while True:
    print('sleep')
    time.sleep(5)
    r = sock.recv()
    print(f'r = {r}')
    break
sock.close()






    
    
            
        
        
