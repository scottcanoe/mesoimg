#import socket
#import numpy as np

#HOST = 'localhost'
#PORT = 7000

#msg = b'Hello, world!'

#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #s.connect((HOST, PORT))
    #s.sendall(msg)
    #data = s.recv(1024)
    #print('Received: {}'.format(data))

from urllib.parse import *
import psutil
procs = [p for p in psutil.process_iter()]
#procs = [p for p in psutil.process_iter() if p.pid == 818]