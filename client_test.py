from pathlib import Path
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
logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(filename=logfile, level=logging.DEBUG)


__all__ = [
    'MesoClient',
]


class MesoClient:



    def __init__(self, host: str = 'pi-meso.local'):

        logging.info('Initializing client.')

        # Networking.
        self._host = host
        self.sockets = {}
        self.pollers = {}
        self.connect()

        # Conccurrency.
        self.threads = {}


    def connect(self):

        logging.info('Opening connections.')

        # Connect to server.
        ctx = zmq.Context.instance()
        cmd = ctx.socket(zmq.REQ)
        cmd.connect(f'tcp://{self._host}:{Ports.COMMAND}')

        # Setup a poller.
        cmd_poller = zmq.Poller()
        cmd_poller.register(cmd, zmq.POLLIN | zmq.POLLOUT)

        # Store sockets and poller(s), attributes.
        self.context = ctx
        self.cmd = cmd
        self.cdm_poller = cmd_poller
        self.sockets['cmd'] = cmd
        self.pollers['cmd'] = cmd_poller


    def disconnect(self):

        for name, sock in self.sockets.items():
            sock.close()
            poller = self.pollers.get(name, None)
            if poller:
                poller.unregister(sock)
        self.sockets.clear()
        self.pollers.clear()


    def close(self):
        self.disconnect()


    def send(self, req: Dict) -> None:
        """
        Final method in pipeline for sending requests to the server.
        """
        logging.debug(f'Sending request: {req}')
        self.cmd.send_json(req)


    def send_get(self, key: str) -> Any:
        """
        Main gateway for retrieving attributes from the server's side. The server
        instance is the target (implicitly). Use this method to ensures that that the
        request is well-formed. Provided as a convenience.

        Get requests have the following structure:
          - 'action' : str     Aalways 'get'.
          - 'key' : str      Name of attribute to set.

        """
        req = {'action' : 'get',
               'key'    : key}
        return self.send(req)


    def send_set(self, key: str, val: Any) -> Any:
        """
        Main gateway for setting attributes on the server side. The server instance
        is the target (implicitly). Use this method to ensures that that the request
        is well-formed. Provided as a convenience.

        Set requests have the following structure:
          - 'action' : str     Always 'set'.
          - 'key' : list     Name of attribute to set.
          - 'val' : dict     Attribute's new value.

        """

        req = {'action' : 'set',
               'key' : key,
               'val' : val}
        return self.send(req)


    def send_call(self, key: str, *args, **kw) -> Any:

        """
        Main gateway for calling methods on the server side. The server instance
        is the target (implicitly). Use this method to ensures that that the request
        is well-formed. Provided as a convenience.

        Call requests have the following structure:
          - 'action' : str   Always 'call'.
          - 'key' : str      Name of callable.
          - 'args' : list    A possibly empty list of positional arguments.
          - 'kw' : dict      A possibly empty dictionary of keyword arguments.

        """

        req = {'action' : 'call',
               'key'  : key,
               'args' : args,
               'kw'   : kw}
        return self.send(req)



c = MesoClient()
