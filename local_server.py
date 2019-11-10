import io
import socket
import numpy as np

HOST = 'localhost'  # This must be left blank!!!
PORT = 7000


arr = np.zeros([480, 640], dtype='u1')
a = np.arange(9, dtype='u1')
a = np.reshape(a, [3, 3])

a = np.array([[1, 2], [3, 4]], dtype='u1')
b = np.ravel(a)
c = memoryview(a)
d = memoryview(b)

buf = io.BytesIO()
#buf.write(b)

#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #s.bind((HOST, PORT))
    #s.listen()
    #conn, addr = s.accept() # Blocking?
    #with conn:
        #print('Connected by: {}'.format(addr))
        #while True:
            #data = conn.recv(1024)
            #if not data:
                #break
            #print('Received: {}'.format(data))
            #conn.sendall(b'Message received.')
            
               
    
