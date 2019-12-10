import os
import sys
from threading import Event, Lock, Thread
import time
from time import perf_counter as clock
from typing import Any, Dict, List, Union
import glom
import numpy as np
import zmq
from mesoimg.app import userdir, find_from_procinfo, kill_from_procinfo, Ports
from mesoimg.common import *
from mesoimg.camera import *
from mesoimg.messaging import *
from mesoimg.outputs import *
import psutil


import logging
logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(filename=logfile, level=logging.DEBUG)


__all__ == [
    'MesoServer',
]



class MesoServer:


    _running: bool = False


    def __init__(self,
                 host: str = '*',
                 start: bool = True,
                 ):

        logging.info('Initializing server.')

        # Networking.
        self.sockets = {}
        self.pollers = {}
        self.bind()

        # Conccurrency.
        self.lock = Lock()
        self.threads = {}
        self._terminate = Event()

        if start:
            self.run()


    #--------------------------------------------------------------------------#
    # Main event loop


    def bind(self):

        logging.info('Opening network connections.')

        # Open client connection.
        ctx = zmq.Context.instance()
        cmd = ctx.socket(zmq.REP)
        cmd.bind(f'tcp://*:{Ports.CONTROL}')

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


    def close(self) -> str:

        # Close network connections, kill threads, etc.
        self._terminate.set()
        self.disconnect()
        for name, thread in self.threads.items():
            pass




    def run(self):
        """


        get:  {'action' : 'get',
               'target' : 'cam',
               'key'    : 'exposure_speed'}

        set:  {'action' : 'set',
               'target' : 'cam',
               'key'    : 'exposure_mode',
               'val'    : 'off'}

        call: {'action' : 'call',
               'target' : '',
               'fn'    : 'close',
               'args'   : [5.0],
               'kw'     : {'some_key' : 55})}


        """

        if self._running:
            msg = 'Client is already running.'
            logging.error(msg)
            raise RuntimeError(msg)
        self._running = True

        # Alias
        cmd = self.sockets['client']
        poller = self.sockets['cmd']
        poller_timout = 1 * 1000

        while not self._terminate.is_set():

            # Check for stdin.
            line = read_stdin().strip()
            if line:
                self._handle_stdin(line)
                continue

            # Check for client requests.
            poll_result = dict(poller.poll(poller_timout))
            if cmd in poll_result and cmd[poll_result] == zmq.POLLIN:
                self._handle_cmd()

        self._running = False


    #--------------------------------------------------------------------------#
    # Sending methods


    def send(self, rep: Dict) -> None:
        """
        Return through here to utilize verbosity.
        """
        logging.debug(f'Sending reply: {rep}')
        self.client_sock.send_json(rep)
        time.sleep(0.005)


    def send_return(self, val: Any = None) -> None:

        rep = {'type' : 'return', 'val' : val}
        self.send(rep)


    def send_error(self, exc: Exception) -> None:

        msg = str(repr(exc))
        print('ERROR: ' + msg, flush=True)
        rep = {'type' : 'error', 'val' : msg}
        self.send(rep)


    #--------------------------------------------------------------------------#
    # Receiving methods


    def _handle_cmd(self):

        logging.debug(f'Received request: {req}', flush=True)
        msg = self.cmd.recv_json()

        #rtype = req.get('type', None)
        #if rtype not in self._request_handlers.keys():
            #self.send_error(RuntimeError(f'invalid request type {rtype}.'))
        time.sleep(0.005)
        self.cmd.send_json({'return' : 'hello there'})

        #self._request_handlers[rtype](req)



    def _handle_get(self, req: Dict) -> None:
        """
        Handle a get request from the client.
        """
        try:
            self.send_return(glom.glom(self, req['key']))
        except Exception as exc:
            self.send_error(exc)


    def _handle_set(self, req: Dict) -> None:
        """
        Handle a set request from the client.
        """
        try:
            glom.assign(self, req['key'], req['val'])
            self.send_return()
        except Exception as exc:
            self.send_error(exc)


    def _handle_call(self, req: Dict) -> None:

        try:
            fn = glom.glom(self, req['key'])
            val = fn(*req['args'], **req['kw'])
            self.send_return(val)
        except Exception as exc:
            self.send_error(exc)


    def _handle_stdin(self, chars: str) -> None:
        chars = 'self.' + chars if not chars.startswith('self.') else chars
        try:
            exec(chars)
        except Exception as exc:
            print(repr(exc))


    #--------------------------------------------------------------------------#
    # etc.

if __name__ == '__main__':

    s = MesoServer()
    try:
        s.run()
    except:
        s.close()
