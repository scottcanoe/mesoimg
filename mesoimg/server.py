import datetime
import os
import sys
from threading import Condition, Event, Lock, Thread
import time
from time import perf_counter as clock
import traceback
from typing import Any, Dict, List, Optional, Union
import glom
import numpy as np
import zmq
from mesoimg.app import userdir, find_from_procinfo, kill_from_procinfo, Ports
from mesoimg.command_line import *
from mesoimg.common import *
#from mesoimg.camera import *
from mesoimg.messaging import *
import psutil


import logging
# logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(level=logging.DEBUG)


__all__ = [
    'MesoServer',
]




class MesoServer:


    _running: bool = False


    def __init__(self, host: str = '*'):

        logging.info('Initializing server.')

        # Networking.
        self.ctx = zmq.Context()
        self.sockets = {}

        # Conccurrency.
        self.lock = Lock()
        self.threads = {}
        self._stop_requested = Event()
        self._stopped = Event()

        self._init()


    #--------------------------------------------------------------------------#
    # Main event loop


    def _init(self):

        self._stop_requested.clear()
        self._stopped.clear()

        self._running = False
        self._terminate = False
        self._exit = False

        # Set up auxilliary namespace for command line usage.
        ns = {}
        ns['echo'] = self.echo
        ns['run'] = self.run
        ns['close'] = self.close
        ns['exit'] = self.close

        self._ns = ns
        self._init_sockets()
        self.start_console()


    def _init_sockets(self) -> None:

        logging.info('Initializing sockets.')

        # Open client connection.
        ctx = zmq.Context.instance()
        com = ctx.socket(zmq.REP)
        com.bind(f'tcp://*:{Ports.COMMAND}')
        poller = zmq.Poller()
        poller.register(com, zmq.POLLIN | zmq.POLLOUT)
        setsockattr(com, 'poller', poller)

        # Store sockets and poller(s), attributes.
        self.com = com
        self.sockets['com'] = com



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
        com = self.sockets['com']
        poller = getsockattr(com, 'poller')
        com_timout = 0.1 * 1000

        logging.info('Starting event loop.')
        write_stdout('> ')
        while not self._terminate:

            # Check for client requests.
            socks = dict(poller.poll(com_timout))
            if socks.get(com, None) == zmq.POLLIN:
            #if com in socks and socks[com] == zmq.POLLIN:
                self.handle_com()

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
        self._init_sockets()



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


    def handle_com(self) -> None:
        """
        Read the client's request, and figure out what to do with it.
        """
        req = self.com.recv_json()
        logging.debug(f'Received request: {req}')

        # Small validity check.
        action = req.get('action', None)
        if action not in ('get', 'set', 'call', 'exec'):
            error = repr(RuntimeError(f'Unsupported action: {action}'))
            resp = dict(action=action, stdout='', error=error)
            self.com.send_json(resp)
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
        self.com.send_json(resp)


    def start_console(self) -> None:

        now = str(datetime.datetime.now())
        parts = now.split(':')
        now = ':'.join(parts[0:2])

        msg = f'MesoServer ({now}) pid = {os.getpid()}'
        print(msg)
        write_stdout('> ')


    #--------------------------------------------------------------------------#
    # etc.

    def echo(self, val=None):
        return val

    def open_camera(self):
        pass
        #from picamera import PiCamera

    def close_camera(self):
        pass




if __name__ == '__main__':


    server = MesoServer()
    server.run()
