import multiprocessing as mp
import sys
import time
import zmq
from mesoimg.messaging import *

from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession


HOST = 'localhost'
PORT = 6001
topic = ''


def recv() -> str:
    """
    Send bytes with a socket.
    """

    # Strip topic if it exists.
    topic = sock.recv_string()
    msg = sock.recv_string()
    return topic, msg


def close() -> None:
    sock.close()
    zmq.Context.instance().term()
    sys.exit()


from threading import Thread
from queue import Queue

PORT = 7018


class Client(Thread):



    def __init__(self):

        super().__init__()

        ctx = zmq.Context.instance()
        self.sock = create_socket('rep', poller=True, timeout=0.01, context=ctx)
        self.sock.bind(f'tcp://*:{PORT}')
        self.terminate = False

    def run(self):

        sock = self.sock
        poller = sock.poller
        timeout = sock.timeout
        timeout = timeout * 1000 if timeout is not None else None

        print('starting client', flush=True)
        while not self.terminate:
            poll_result = dict(poller.poll(timeout))
            if can_recv(sock, poll_result):
                msg = sock.recv_string()
                new_msg = f'STAMP: {msg}'
                print(new_msg, flush=True)
                time.sleep(0.05)
                sock.send_string(new_msg)

        #sock.close()


ps = PromptSession('> ')

ctx = zmq.Context.instance()
sock = create_socket('req', context=ctx)
sock.connect(f'tcp://127.0.0.1:{PORT}')

client = Client()
time.sleep(1)
client.start()

s1, p1 = sock, sock.poller
s2, p2 = client.sock, client.sock.poller

def run():

    terminate = False
    while not terminate:
        try:
            line = ps.prompt()
            if line == 'q':
                terminate = True
                print('terminate requested', flush=True)
                continue
            sock.send_string(line)
            time.sleep(0.01)
            ret = sock.recv_string()
            print(f'returned: {ret}', flush=True)
            continue

        except KeyboardInterrupt:
            terminate = True

        except Exception as exc:
            print(f'Exception occurred: {repr(exc)}', flush=True)


def close():
    p1.unregister(s1)
    s1.close()
    p2.unregister(s2)
    s2.close()
    ctx.term()


#cs = client.sock
#client.terminate = True
#time.sleep(0.1)
#sock.close()



