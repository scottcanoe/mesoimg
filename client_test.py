from pathlib import Path
import sys
from threading import Condition, Event, Lock, Thread
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import zmq
from mesoimg.app import userdir, find_from_procinfo, kill_from_procinfo, Ports
from mesoimg.common import *
from mesoimg.inputs import *
from mesoimg.messaging import *
from mesoimg.parsing import *
from mesoimg.requests import *

import logging
#logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(level=logging.DEBUG)


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


    def _init_socket(self) -> None:

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
        sock = self.socket
        data = req.asdict() if isinstance(req, Request) else req
        sock.send_json(data)
        return self.recv(**kw)


    def recv(self, **kw) -> Response:
        timeout = kw.get('timeout', self.timeout)
        info = dict(self.in_poller.poll(timeout))
        if self.socket in info:
            data = self.socket.recv_json()
            resp = Response(**data)
            logging.debug(f'Received: {resp}')
            return resp

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

        # Connect command socket.
        self._init()

        # Conccurrency.
        self._threads = {}


    def _init(self) -> None:
        logging.info('Initializing client.')
        self._init_sockets()


    def _init_sockets(self) -> None:

        logging.info('Initializing sockets.')

        # Connect to server.
        self.ctx = zmq.Context()
        self.com = CommandSocket(zmq.REQ, self._host, Ports.COMMAND)

        # Store sockets.
        self.sockets['com'] = self.com


    def close(self):
        logging.info('closing client')
        self.close_sockets()


    def close_sockets(self):
        logging.info('closing sockets')
        for name, sock in self.sockets.items():
            sock.close()


    def reset(self):
        self.close()
        self._init()


    def reset_sockets(self):
        self.close_sockets()
        time.sleep(0.5)
        self._init_sockets()



if __name__ == '__main__':


    def close():
        return client.close()

    def reset():
        client.reset()

    def close_sockets():
        return client.close_sockets()

    def reset_sockets():
        client.reset_sockets()

    def exit():
        client.close()
        sys.exit()

    client = MesoClient(host='127.0.0.1')
    com = client.com


