import sys
from queue import Queue
from mesoimg.messaging import *
import zmq

def init_pub():
    sock = ctx.socket(zmq.PUB)
    sock.bind(BIND)
    poller = zmq.Poller()
    poller.register(sock)
    return sock, poller

def init_sub():
    sock = ctx.socket(zmq.SUB)
    sock.subscribe(topic.encode())
    sock.rcvtimeo = 10 * 1000
    sock.connect(CONNECT)
    poller = zmq.Poller()
    poller.register(sock)
    return sock, poller

def init_pub2():

    q = Queue()
    sock = Publisher(q, send_string, '')
    sock.bind(BIND)
    return sock, None

def init_sub2():
    q = Queue()
    sock = Subscriber(recv_string, q, '')
    sock.connect(f'tcp://{HOST}:{PORT}')
    return sock, None


def poll(timeout=0):
    return dict(poller.poll(0))

def send(msg):
    sock.send_string(msg)

def recv():
    sock.recv_string()

def pub(msg):
    sock.send_string(topic, zmq.SNDMORE)
    sock.send_string(msg)

def sub():
    sock.recv_string()
    return sock.recv_string()

def close():
    sock.close()
    if isinstance(sock, SocketThread):
        sock.join()
    ctx.term()

def exit():
    try:
        close()
    except:
        pass
    sys.exit()


if __name__ == '__main__':

    HOST = '127.0.0.1'
    #HOST = 'pi-meso.local'
    PORT = 9000
    BIND = f'tcp://*:{PORT}'
    CONNECT = f'tcp://{HOST}:{PORT}'
    topic = ''
    ctx = zmq.Context()

    args = sys.argv[1:]
    init_func = locals()[args[0]]
    init = lambda : init_func()

    sock, poller = init()
    sock.start()
