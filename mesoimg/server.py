from queue import Queue
from threading import Event, Lock, Thread
import time
from time import perf_counter as clock
from typing import Dict
import numpy as np
import zmq
from mesoimg.common import *
from mesoimg.camera import *
from mesoimg.outputs import *



class MesoServer:


    def __init__(self):

        self.sockets = {}
        self.threads = {}

        self.ctx = zmq.Context()
        self.cmd_sock = self.ctx.socket(zmq.PAIR)
        self.cmd_sock.bind(f'tcp://*:{Ports.COMMAND}')
        self.cmd_poller = zmq.Poller()
        self.cmd_poller.register(self.cmd_sock, zmq.POLLIN | zmq.POLLOUT)
        self.cmd_timeout = 1.0
        self.sockets['cmd'] = self.cmd_sock

        # Open camera, and prepare to publish frames.
        self.cam = Camera()
        self.frame_publisher = FramePublisher(self.ctx, self.cam)
        self._terminate = False
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

        print('Starting server.', flush=True)

        # Alias
        sock = self.cmd_sock
        poller = self.cmd_sock.poller
        timeout = self.cmd_timeout

        self._terminate = False
        while not self.terminate:

            # Poll for request from client.
            ready = dict(poller.poll(timeout))
            if not (sock in ready and ready[sock] == zmq.POLLIN):
                continue
            req = sock.recv_json()

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



    def close(self) -> str:
        print('Closing server.')
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
    # client messaging

    """
    type: 'return', 'error'

    """

    def _return_val(self, val: Any = '', info: str = '') -> None:

        val = '' if val is None else val
        rep = {'type' : 'val',
               'val' : val,
               'info' : info}
        self.cmd_sock.send_json(rep)
        time.sleep(0.05)


    def _return_error(self, exc: Exception, info: Any = '') -> None:

        msg = str(repr(exc))
        print(msg, flush=True)
        rep = {'type' : 'error',
               'error' : msg,
               'info' : info}
        self.cmd_sock.send_json(rep)
        time.sleep(0.05)


    #--------------------------------------------------------------------------#
    #

    def _cleanup(self):
        """
        Prepare to close/exit the server.
        """
        for sock in self.sockets.values():
            sock.close()
        for thread in self.threads.values():
            thread.close()
        self.cam.close()







