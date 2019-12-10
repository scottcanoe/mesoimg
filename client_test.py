from pathlib import Path
import sys
from threading import Condition, Event, Lock, Thread
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from superjson import json
import zmq
from mesoimg.app import userdir, find_from_procinfo, kill_from_procinfo, Ports
from mesoimg.common import *
from mesoimg.inputs import *
from mesoimg.messaging import *
from mesoimg.parsing import *
from mesoimg.requests import *

import logging
#logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(level=logging.INFO)


__all__ = [
    'MesoClient',
]



class CommandSocket:



    def __init__(self,
                 sock_type: int,
                 host: str,
                 port: Union[int, str],
                 binds: bool = False,
                 timeout: float = 10.0,
                 ):

        self.socket = None
        self.poller = None
        self._sock_type = sock_type
        self._host = host
        self._port = port
        self._binds = binds
        self.timeout = timeout

        self._init_socket()

        self.last = None # Last response returned.



    def _init_socket(self) -> None:

        logging.info('Connecting command socket.')
        # Create and bind the socket.
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(self._sock_type)
        addr = f'tcp://{self._host}:{self._port}'
        if self._binds:
            self.socket.bind(addr)
        else:
            self.socket.connect(addr)

        # Setup a poller.
        self.poller = zmq.Poller()
        self.poller.register(self.socket)
        self.in_poller = zmq.Poller()
        self.in_poller.register(self.socket, zmq.POLLIN)


    def close(self):
        self.socket.close()
        for p in [self.poller, self.in_poller]:
            try:
                p.unregister(self.socket)
            except:
                pass

    def reset(self):

        self.close()
        time.sleep(0.005)
        self._init_socket()


    def send(self, req: Request, **kw) -> Optional[Response]:

        """
        Send a request to the server, and wait for a response.
        """

        logging.debug(f'Sending request: {req}')
        data = req.asdict() if isinstance(req, Request) else req
        send_json(self.socket, data)
        return self.recv(**kw)


    def recv(self, **kw) -> Response:
        timeout = kw.get('timeout', self.timeout) * 1000
        info = dict(self.in_poller.poll(timeout))
        if self.socket in info:
            data = recv_json(self.socket)
            resp = Response(**data)
            self.last = resp
            logging.debug(f'Received: {resp}')

            if resp.error:
                pprint(resp.error)
                return resp
            if resp.stdout:
                pprint(resp.stdout)
            return resp.result


        msg  = 'No message received within timeout period. '
        msg += 'Socket is still in receiving state.'
        logging.warning(msg)


    def get(self, key, **kw) -> Optional[Response]:
        return self.send(Get(key), **kw)


    def set(self, key, val, **kw) -> Optional[Response]:
        return self.send(Set(key, val), **kw)


    def call(self, key, *args, **kw) -> Optional[Response]:
        timeout = kw.pop('timeout', self.timeout)
        return self.send(Call(key, args, kw), timeout=timeout)


    def exec(self, text, **kw) -> Optional[Response]:
        timeout = kw.pop('timeout', self.timeout)
        return self.send(Exec(text), **kw)



class MesoClient:



    def __init__(self, host: str = 'pi-meso.local'):

        # Networking.
        self._host = host
        self.sockets = {}
        self.cmd = None

        # Connect command socket.
        self._init()

        # Conccurrency.
        self.threads = {}


    def _init(self, reset: bool = False) -> None:
        logging.info(f'Initializing client (reset={reset})')
        self._init_sockets(reset=reset)


    def _init_sockets(self, reset: bool = False) -> None:

        logging.info(f'Initializing sockets (reset={reset})')

        # Connect to server.
        self.ctx = zmq.Context()
        if reset and self.cmd:
            self.cmd.reset()
        else:
            self.cmd = CommandSocket(zmq.REQ, self._host, Ports.COMMAND)
            self.sockets['cmd'] = self.cmd


    def close(self):
        logging.info('closing client')
        self.close_sockets()


    def close_sockets(self):
        logging.info('closing sockets')
        for name, sock in self.sockets.items():
            sock.close()


    def reset(self):
        self.close()
        self._init(reset=True)


    def reset_sockets(self):
        self.close_sockets()
        time.sleep(0.5)
        self._init_sockets(reset=True)




