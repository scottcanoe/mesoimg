import socket


HOST = ''  # This must be left blank!!!
PORT = 7000


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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
            
               
    
