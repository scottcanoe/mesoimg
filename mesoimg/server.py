from queue import Queue
from threading import Event, Lock, Thread
import time
from time import perf_counter as clock
from typing import Any, Dict, List, Union
import numpy as np
import zmq
from mesoimg.common import *
from mesoimg.camera import *
from mesoimg.outputs import *



class MesoServer:


    verbose = False

    def __init__(self, start: bool = True):

        self.sockets = {}
        self.threads = {}

        self.ctx = zmq.Context()
        self.cmd_sock = self.ctx.socket(zmq.REP)
        self.cmd_sock.bind(f'tcp://*:{Ports.COMMAND}')
        self.cmd_poller = zmq.Poller()
        self.cmd_poller.register(self.cmd_sock, zmq.POLLIN | zmq.POLLOUT)
        self.cmd_timeout = 1.0
        self.sockets['cmd'] = self.cmd_sock

        # Open camera, and prepare to publish frames.
        self.cam = Camera()
        self.frame_publisher = FramePublisher(self.ctx, self.cam)
        self.threads['frame_publisher'] = self.frame_publisher

        self._terminate = False

        if start:
            self.run()




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

        print('Server ready.', flush=True)

        # Alias
        sock = self.cmd_sock
        poller = self.cmd_poller
        timeout = self.cmd_timeout

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
            if self.verbose:
                print(f'Received request: {req}', flush=True)

            # Check action.
            if 'action' not in req:
                self._return_error(RuntimeError('no action in request.'))
                continue
            action = req['action']
            if action not in ('get', 'set', 'call'):
                self._return_error(RuntimeError(f'invalid action: {action}'))
                continue

            # Check target.
            if 'target' not in req:
                self._return_error(RuntimeError('no target in request.'))
                continue
            target = req['target']
            if target not in ('cam', 'server'):
                self._return_error(RuntimeError(f'invalid target: {target}'))
                continue
            target = self.cam if target == 'cam' else self

            # Check key.
            if 'key' not in req:
                self._return_error(RuntimeError('no key in request.'))
                continue
            key = req['key']


            # Handle get request.
            if action == 'get':
                self._handle_get(target, key)
                continue

            # Handle set request.
            if action == 'set':
                if 'val' not in req:
                    self._return_error(RuntimeError('no value in set request.'))
                    continue
                self._handle_set(target, key, req['val'])
                continue

            # Handle call request.
            args, kw = req.get('args', []), req.get('kw', {})
            self._handle_call(target, key, args, kw)
            continue


        # Finally, close down.
        self._cleanup()
        print('Server closed.')



    def close(self) -> str:
        self._terminate = True


    def start_preview(self):
        self.cam_thread = Thread(target=self.cam.start_preview)
        self.cam_thread.start()
        return 0


    def stop_preview(self):
        self.cam._stop = True
        return 0

    def start_recording(self, duration):
        self.cam_thread = Thread(target=self.cam.start_recording,
                                 args=(duration,))
        self.cam_thread.start()
        return 0

    def stop_recording(self):
        self.cam._stop = True
        return 0


    #--------------------------------------------------------------------------#
    # get/set/call handlers


    def _handle_get(self,
                    target: Union['MesoServer', Camera],
                    key: str,
                    ) -> None:
        try:
            self._return_val(getattr(target, key))
        except Exception as exc:
            self._return_error(exc)


    def _handle_set(self,
                    target: Union['MesoServer', Camera],
                    key: str,
                    val: Any,
                    ) -> None:

        try:
            self._return_val(setattr(target, key, val))
        except Exception as exc:
            self._return_error(exc)


    def _handle_call(self,
                     target: Union['MesoServer', Camera],
                     key: str,
                     args: List,
                     kw: Dict,
                     ) -> None:

        try:
            fn = getattr(target, key)
            self._return_val(fn(*args, **kw))
        except Exception as exc:
            self._return_error(exc)
            return


    #--------------------------------------------------------------------------#
    # stdin handlers


    def _handle_stdin(self, line: str) -> None:
        line = 'self.' + line
        exec(line)


    #--------------------------------------------------------------------------#
    # client messaging

    """
    type: 'return', 'error'

    """

    def _reply(self, rep: Dict) -> None:
        if self.verbose:
            print(f'Sending reply: {rep}')
        self.cmd_sock.send_json(rep)
        time.sleep(0.05)


    def _return_val(self, val: Any = '') -> None:

        val = '' if val is None else val
        rep = {'type' : 'return', 'val' : val}
        self._reply(rep)


    def _return_error(self, exc: Exception) -> None:

        msg = str(repr(exc))
        print('ERROR: ' + msg, flush=True)
        rep = {'type' : 'error', 'val' : msg}
        self._reply(rep)


    #--------------------------------------------------------------------------#
    #

    def _cleanup(self):
        """
        Prepare to close/exit the server.
        """
        print('Closing sockets.')
        for sock in self.sockets.values():
            sock.close()
        print('Closing threads.')
        for thread in self.threads.values():
            thread.close()
        self.cam.close()
        time.sleep(1.5)
        self.ctx.term()


