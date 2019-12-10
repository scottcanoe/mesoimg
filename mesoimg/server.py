import datetime
import os
from pprint import pprint
import sys
from threading import Event, Lock, Thread
import time
from time import perf_counter as clock
from typing import Any, Dict, List, Union
import glom
import numpy as np
import zmq
from mesoimg.app import userdir, find_from_procinfo, kill_from_procinfo, Ports
from mesoimg.command_line import *
from mesoimg.common import *
from mesoimg.camera import *
from mesoimg.messaging import *
from mesoimg.outputs import *
import psutil


import logging
# logfile = userdir() / 'logs' / 'log.txt'
logging.basicConfig(level=logging.DEBUG)


__all__ = [
    'MesoServer',
]



def start_console() -> None:

    now = str(datetime.datetime.now())
    parts = now.split(':')
    now = ':'.join(parts[0:2])

    msg = f'MesoServer ({now}) pid = {os.getpid()}'
    print(msg)
    prompt()


def prompt() -> None:
    sys.stdout.write('> ')
    sys.stdout.flush()


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

        # Set up auxilliary namespace for command line usage.
        ns = {}
        ns['bind'] = self.bind
        ns['disconnect'] = self.disconnect
        ns['close'] = self.close

        self._ns = ns
        if start:
            self.run()


    #--------------------------------------------------------------------------#
    # Main event loop


    def bind(self):

        logging.info('Opening network connections.')

        # Open client connection.
        ctx = zmq.Context.instance()
        cmd = ctx.socket(zmq.REP)
        cmd.bind(f'tcp://*:{Ports.COMMAND}')

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
        print('')



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
            logging.warning(msg)
            return
        self._running = True

        # Alias
        cmd = self.sockets['cmd']
        poller = self.pollers['cmd']
        poller_timout = 0.1 * 1000

        logging.info('Ready for commands.')

        # Setup CLI.
        start_console()

        while not self._terminate.is_set():

            # Check for stdin.
            if poll_stdin():
                self.handle_stdin()

            # Check for client requests.
            res = dict(poller.poll(poller_timout))
            if cmd in res and res[cmd] == zmq.POLLIN:
                self.handle_cmd()

        self._running = False



    #--------------------------------------------------------------------------#
    # Receiving methods

    def handle_stdin(self) -> None:
        s = read_stdin().rstrip()
        res, out, err = execute(s, globals(), self._ns)


        if err:
            if err.endswith('\n'):
                err = err[:-1]
            print(err)
        elif res:
            pprint(res)

        prompt()
        return


    def handle_cmd(self) -> None:


        req = self.cmd.recv_json()
        logging.debug(f'Received request: {req}')

        # Small validity check.
        action = req.get('action', None)
        if action not in ('get', 'set', 'call'):
            err = repr(RuntimeError(f'Unsupported action: {action}'))
            self.cmd.send_json({'error' : repr(err)})
            time.sleep(0.005)
            return

        try:

            if action == 'get':
                val = glom.glom(self, req['key'])

            elif action == 'set':
                glom.assign(self, req['key'], req['val'])

            else:

                fn = glom.glom(self, req['key'])
                val = fn(*req['args'], **req['kw'])
                self.send_return(val)

            except Exception as exc:
                self.send_error(exc)


        except:
            err = ''.join(traceback.format_exception(*sys.exc_info()))



    #--------------------------------------------------------------------------#
    # Sending methods


    def send_(self, rep: Dict) -> None:
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
    # etc.

if __name__ == '__main__':

    s = MesoServer()
    try:
        s.run()
    except:
        s.close()
