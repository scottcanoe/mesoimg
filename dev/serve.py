import time
import numpy as np
import zmq
from mesoimg.common import *


PORT = 7000



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


data = np.zeros([480, 640], dtype=np.uint8)
for i in range(data.shape[0]):
    data[i] = i


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f'tcp://*:{PORT}')
terminate = False
while not terminate:
    
    print(f'Binding to port {PORT}', flush=True)
    req = socket.recv_string()
    print(f"Received {type(req)}: {req}")
    if req == 'frame':
        send_frame(socket, data, 5, 3.1415)
    elif req == 'quit':
        socket.send_string('quitting')
        terminate = True
        print("Shutting down.")
        time.sleep(0.1)
        socket.close()
    else:
        socket.send_string('huh?')
        
        

