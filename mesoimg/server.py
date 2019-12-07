import sys
from threading import Event, Lock, Thread
import time
from time import perf_counter as clock
from typing import Any, Dict, List, Union
import glom
import numpy as np
import zmq
from mesoimg.common import *
from mesoimg.camera import *
from mesoimg.outputs import *



class MesoServer:


    verbose: bool = False

    _terminate: bool = False


    def __init__(self,
                 context: Optional[zmq.Context] = None,
                 start: bool = False,
                 ):

        print('Initializing MesoServer.')

        if context is None:
            ctx = zmq.Context.instance()

        self.sockets = {}
        self.threads = {}

        self.ctx = ctx
        self.cmd_sock = self.ctx.socket(zmq.REP)
        self.cmd_sock.bind(f'tcp://*:{Ports.COMMAND}')
        self.cmd_poller = zmq.Poller()
        self.cmd_poller.register(self.cmd_sock, zmq.POLLIN | zmq.POLLOUT)
        self.cmd_timeout = 1.0
        self.sockets['cmd'] = self.cmd_sock

        # Set up request handling.
        self._request_handlers = {'get' : self._handle_get,
                                  'set' : self._handle_set,
                                  'call': self._handle_call}

        # Open camera, and prepare to publish frames.
        self.cam = Camera()
        self.frame_publisher = FramePublisher(self.ctx, self.cam)
        self.threads['frame_publisher'] = self.frame_publisher

        self._started = False
        self._running = False
        self._terminate = False

        self.run()


    #--------------------------------------------------------------------------#
    # Main event loop


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
               'key'    : 'close',
               'args'   : [5.0],
               'kw'     : {'some_key' : 55})}


        """

        if self._started:
            print('Server has already run.')
            return

        print('Server ready.', flush=True)

        # Alias
        sock = self.cmd_sock
        poller = self.cmd_poller
        timeout = self.cmd_timeout

        self._started = True
        self._running = True
        self._terminate = False
        while not self._terminate:

            # Check for stdin.
            line = read_stdin().strip()
            if line:
                self._handle_stdin(line)
                continue

            # Check for client requests.
            ready = dict(poller.poll(timeout * 1000))
            if not (sock in ready and ready[sock] == zmq.POLLIN):
                continue
            req = sock.recv_json()
            self._handle_request(req)
            continue


        # Finally, shut everything down neatly.
        self._shutdown()


    def close(self) -> str:
            self._terminate = True


    def _shutdown(self) -> None:
        """Shut everything down neatly."""
        print('Closing sockets.')
        for sock in self.sockets.values():
            sock.close()
        print('Closing threads.')
        for thread in self.threads.values():
            thread.close()
        self.cam.close()
        time.sleep(1.5)
        self.ctx.term()
        self._running = False
        print('Server closed.')


    #--------------------------------------------------------------------------#
    # Sending methods


    def send(self, rep: Dict) -> None:
        """
        Return through here to utilize verbosity.
        """
        if self.verbose:
            print(f'Sending reply: {rep}')
        self.cmd_sock.send_json(rep)
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


    def _handle_request(self, req: Dict) -> None:

        if self.verbose:
            print(f'Received request: {req}', flush=True)

        rtype = req.get('type', None)
        if rtype not in self._request_handlers.keys():
            self.send_error(RuntimeError(f'invalid request type {rtype}.'))

        self._request_handlers[rtype](req)



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


def write_prompt():
    sys.stdout.write('>>> ')


