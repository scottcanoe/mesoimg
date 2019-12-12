import datetime
import os
import sys
from threading import Condition, Event, Lock, RLock, Thread
import time
from time import perf_counter as clock
import traceback
from typing import Any, Dict, List, Optional, Union
import glom
import numpy as np
from superjson import json
import zmq
from mesoimg import *
import psutil


import logging
logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(level=logging.DEBUG)


__all__ = [
    'MesoServer',
]




class MesoServer:


    cam: Optional[Camera] = None
    _running: bool = False
    _terminate: bool = False


    def __init__(self,
                 host: str = '*',
                 context: Optional[zmq.Context] = None,
                 ):

        logging.info('Initializing server.')

        # Networking.
        self.ctx = context if context else zmq.Context.instance()
        self.sockets = {}

        # Conccurrency.
        self.lock = Lock()
        self.rlock = RLock()
        self.threads = {}
        self._stop_requested = Event()
        self._stopped = Event()

        self._init()


    #--------------------------------------------------------------------------#
    # Main event loop


    def _init(self):

        # Set flags.
        self._stop_requested.clear()
        self._stopped.clear()
        self._running = False
        self._terminate = False

        # Open networking.
        self._init_sockets()

        # Set up auxilliary namespace for command line usage.
        ns = {}

        # - server
        ns['server'] = self
        ns['echo'] = self.echo
        ns['run'] = self.run
        ns['close'] = self.close
        ns['exit'] = self.close

        # - camera
        ns['cam'] = self.cam
        ns['open_camera'] = self.open_camera
        ns['close_camera'] = self.close_camera

        # Start the console.
        self._ns = ns
        self.start_console()




    def _init_sockets(self) -> None:

        logging.info('Initializing sockets.')

        # Open client connection.
        ctx = zmq.Context.instance()
        cmd = ctx.socket(zmq.REP)
        cmd.bind(f'tcp://*:{Ports.COMMAND}')
        poller = zmq.Poller()
        poller.register(cmd, zmq.POLLIN | zmq.POLLOUT)
        setsockattr(cmd, 'poller', poller)

        # Store sockets and poller(s), attributes.
        self.cmd = cmd
        self.sockets['cmd'] = cmd



    def close(self) -> None:

        logging.info('Closing server.')

        # Stop the event loop, and wait for it to finish.
        self._terminate = True


    def run(self):

        if self._running:
            print('Already running...')
            return
        self._running = True
        self._terminate = False

        # Alias
        cmd = self.sockets['cmd']
        poller = getsockattr(cmd, 'poller')
        cmd_timout = 0.1 * 1000

        logging.info('Starting event loop.')
        write_stdout('> ')
        while not self._terminate:

            # Check for client requests.
            socks = dict(poller.poll(cmd_timout))
            if socks.get(cmd, None) == zmq.POLLIN:

                self.handle_cmd()

            # Check for stdin.
            if poll_stdin():
                self.handle_stdin()

        logging.info('Event loop stopped.')
        self._running = False
        self.cleanup()



    def cleanup(self) -> None:

        logging.info('Closing sockets.')
        if self._running:
            msg = "Stop event loop before closing sockets."
            logging.warning(msg)
            return msg

        # Wait for event loop to stop.
        for sock in self.sockets.values():
            sock.close()


    def reset(self):
        self.close()
        time.sleep(0.1)
        self._init()



    #--------------------------------------------------------------------------#
    # Receiving methods


    def handle_stdin(self) -> None:
        """
        Ready command line input, and execute it.
        """
        s = read_stdin().rstrip()
        result, stdout, error = execute(s, globals(), self._ns)

        if error:
            if error.endswith('\n'):
                error = error[:-1]
            print(error)
        elif result:
            pprint(result)

        write_stdout('> ')
        return


    def handle_cmd(self) -> None:
        """
        Read the client's request, and figure out what to do with it.
        """
        j_req = self.cmd.recv_string()
        req = json.loads(j_req)
        logging.debug(f'Received request: {req}')

        # Small validity check.
        action = req.get('action', None)
        if action not in ('get', 'set', 'call', 'exec'):
            error = repr(RuntimeError(f'Unsupported action: {action}'))
            resp = dict(action=action, stdout='', error=error)
            self.cmd.send_json(resp)
            return

        result = None
        stdout = ''
        error = ''

        try:

            if action == 'get':
                result = glom.glom(self, req['key'])

            elif action == 'set':
                glom.assign(self, req['key'], req['val'])

            elif action == 'call':
                fn = glom.glom(self, req['key'])
                result = fn(*req['args'], **req['kw'])

            elif action == 'exec':
                _, stdout, error = execute(req['text'], globals(), locals())

        except:
            error = ''.join(traceback.format_exception(*sys.exc_info()))

        resp = dict(result=result, stdout=stdout, error=error)
        logging.debug(f'Returning: {resp}')

        try:
            j_resp = json.dumps(resp)
        except Exception as exc:
            error = ''.join(traceback.format_exception(*sys.exc_info()))
            resp = dict(result=None, stdout='', error=error)
            j_resp = json.dumps(resp)

        self.cmd.send_string(j_resp)


    #--------------------------------------------------------------------------#
    # etc.


    def start_console(self) -> None:

        now = str(datetime.datetime.now())
        parts = now.split(':')
        now = ':'.join(parts[0:2])

        msg = f'MesoServer ({now}) pid = {os.getpid()}'
        print(msg)
        write_stdout('> ')


    def open_camera(self) -> None:

        if self.cam and not self.cam.closed:
            raise RuntimeError("camera already exists and is open.")

        self.cam = Camera()
        self._ns['cam'] = self.cam


    def close_camera(self) -> None:
        self.cam.close()


    def echo(self, val=None):
        return val

    def open_camera(self):
        pass
        #from picamera import PiCamera

    def close_camera(self):
        pass


