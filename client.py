import socket

#HOST = 'localhost'
#HOST = '169.254.77.134'
HOST = '169.254.77.134'
PORT = 7000

msg = b'Local message'

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect((HOST, PORT))

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(msg)
    data = s.recv(1024)
    print('Received: {}'.format(data))

    
