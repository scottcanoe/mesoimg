"""
Module documentation.
"""
import io
import socket
import numpy as np
import zmq

HOST = ''  # This must be left blank when network with other computers!!!
PORT = 7000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept() # Blocking?
with conn:
    print('Connected by: {}'.format(addr))
    while True:
        data = conn.recv(1024)
        if not data:
            break
        print('Received: {}'.format(data))
        conn.sendall(b'Message received.')
s.close()
