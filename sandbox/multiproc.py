import logging
import os
from pathlib import Path
import io
import time
import multiprocessing as mp
from queue import Empty
import zmq
from mesoimg import *


filename = Path.home() / 'log.txt'
logging.basicConfig(filename=filename, level=logging.DEBUG)

PORT = 8004

def run_proc(in_q, out_q, terminate):

    log = logging.getLogger('proc')

    log.info('started')

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f'tcp://*:{PORT}')

    while not terminate.wait(0.1):
        try:
            val = in_q.get(timeout=0.1)

        except Empty:
            continue
        out_q.put(val)
        msg = f'val={val}'
        sock.send_string(msg)

    sock.close()


q =  mp.Queue()
out =  mp.Queue()
terminate = mp.Event()

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect(f'tcp://localhost:{PORT}')
sock.subscribe(b'')
sock.rcvtimeo = 1000
p = mp.Process(target=run_proc, args=(q, out, terminate))
p.start()

time.sleep(1)
q.put(0)
time.sleep(1)
q.put(1)


