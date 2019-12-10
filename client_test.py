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

import logging
logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(filename=logfile, level=logging.DEBUG)


__all__ = [
    'MesoClient',
]




class MesoClient:



    def __init__(self,
                 host: str = 'pi-meso.local',
                 start: bool = False,
                 ):

        logging.info('Initializing client.')

        # Networking.
        self.sockets = {}
        self.pollers = {}
        self.connect()

        # Conccurrency.
        self.threads = {}

        if start:
            self.run()


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


    #------------------------------------------------------------------------------------#
    # Main event loop

    #def run(self):
        #"""
        #Event loop that read user input from stdin, interprets them as requests,
        #sends requests, and finally handles replies.

        #The target object (the 'base' of the namespace) is implicitly the client
        #instance itself.


        #"""
        #if self._running:
            #msg = 'Client is already running.'
            #logging.error(msg)
            #raise RuntimeError(msg)

        ## Alias
        #server = self._sockets['server']
        #shell = self._sockets['shell']
        #poller = self._poller
        #timeout = self.polling_timeout * 1000

        #while not self._terminate:
            #poll_result = dict(poller.poll(timeout))

            ## Check for responses from the server.
            #if can_recv(server, poll_result):
                #rep = server.read_json()
                #self._handle_server_reply(rep)

            ## Check for input from the shell/terminal.
            #if can_recv(shell, poll_result):
                #req = shell.read_string()
                #self._handle_shell_request(req)


        ## Finally, shut everything down neatly.
        #self._shutdown()


    #def _handle_server_reply(self, rep: Dict) -> None:
        #"""
        #Replies: print to screen?
        #"""




    #def _handle_shell_request(self, req: str) -> None:

        #"""
        #Read and parse the request
        #"""
        #target = infer_target(req)
        #if target in ('cam.', 'server.'):
            #self._sockets['server'].send_string(req)
            #time.sleep(0.01)

        #efun = infer_efun(req)



c = MesoClient()
