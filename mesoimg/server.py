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

        self.cam = Camera()

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

        # Alias
        sock = self.cmd_sock
        poller = self.cmd_sock.poller
        timeout = self.cmd_timeout


        self._terminate = False

        print('Ready for commands.', flush=True)
        while not self.terminate:
            ready = dict(poller.poll(timeout))
            if not (sock in ready and ready[sock] == zmq.POLLIN):
                continue
            req = sock.recv_json()

            # Check action.
            if 'action' not in req:
                self.send_exception(RuntimeError('no action in request.'))
                continue
            action = req['action']
            if action not in ('get', 'set', 'call'):
                self.send_exception(RuntimeError(f'invalid action: {action}'))
                continue

            # Check target.
            if 'target' not in req:
                self.send_exception(RuntimeError('no target in request.'))
                continue
            target = req['target']
            if target not in ('cam', 'server'):
                self.send_exception(RuntimeError(f'invalid target: {target}'))
                continue
            target = self.cam if target == 'cam' else self

            # Check key.
            if 'key' not in req:
                self.send_exception(RuntimeError('no key in request.'))
                continue
            key = req['key']


            # Handle get request.
            if action == 'get':
                self._handle_get(target, key)
                continue

            # Handle set request.
            if action == 'set':
                if 'val' not in req:
                    self.send_exception(RuntimeError('no value in set request.'))
                    continue
                self._handle_set(target, key, req['val'])
                continue

            # Handle call request.
            args, kw = req.get('args', []), req.get('kw', {})
            self._handle_call(target, key, args, kw)
            continue




    def close(self):

        self._terminate = True
        time.sleep(self.cmd_timeout + 1.0)

        for name, sock in self.sockets.items():
            if name != 'cmd':
                sock.close()
        for thread in self.threads.values():
            thread.close()
        self.cam.close()

        self.cmd_sock.send_json()



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
            self._return_value(getattr(target, key))
        except Exception as exc:
            self._return_exception(exc)


    def _handle_set(self,
                    target: Union['MesoServer', Camera],
                    key: str,
                    val: Any,
                    ) -> None:

        try:
            self._return_value(setattr(target, key, val))
        except Exception as exc:
            self._return_exception(exc)


    def _handle_call(self,
                     target: Union['MesoServer', Camera],
                     key: str,
                     args: List,
                     kw: Dict,
                     ) -> None:

        try:
            fn = getattr(target, key)
            self._return_value(fn(*args, **kw))
        except Exception as exc:
            self._return_exception(exc)
            return


    #--------------------------------------------------------------------------#
    # client messaging

    """
    type: 'return', 'error'

    """

    def _return_value(self, val: Any = '', info: str = '') -> None:

        val = '' if val is None else val
        rep = {'class' : 'value',
               'value' : val,
               'info' : info}
        self.cmd_sock.send_json(rep)
        time.sleep(0.05)


    def _return_exception(self, exc: Exception, info: Any = '') -> None:

        exc_string = str(repr(exc))
        print(exc_string, flush=True)
        rep = {'class' : 'exception',
               'exception' : exc_string,
               'info' : info}
        self.cmd_sock.send_json(rep)
        time.sleep(0.05)













