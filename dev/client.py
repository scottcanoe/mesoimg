import socket

# Use this method when connecting from a networked computer.
HOST_NAME = 'pi-meso.local' # Don't forget to add '.local' !!!
HOST = socket.gethostbyname(HOST_NAME)
PORT = 7000

msg = b'Hello, world!'
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #s.connect((HOST, PORT))
    #s.sendall(msg)
    #data = s.recv(1024)
    #print('Received: {}'.format(data))


