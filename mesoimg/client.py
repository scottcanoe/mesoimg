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


#class MesoClient:

    #_host: str
    #_ctx: zmq.Context
    #_sockets: Dict[str, zmq.Socket]
    #_threads: Dict[str, Thread]
    #_server: zmq.Socket

    #_started: bool
    #_running: bool
    #_terminate: bool
    #_stopped: bool

    #polling_timeout = 0.01


    #def __init__(self,
                 #host: str = 'pi-meso.local',
                 #start: bool = True,
                 #):

        #logging.info('Initializing server.')

        #self._host = host
        #self._context = zmq.Context.instance()
        #self._sockets = {}
        #self._threads = {}

        ## Setup conditions, events, etc.
        ##self.frame_received = Condition()
        ##self.frame_q = Queue(maxsize=30)
        ##self.status_q = Queue(maxsize=30)

        #self._started = False
        #self._running = False
        #self._terminate = False
        #self._stopped = False

        #if start:
            #self.start()


    #def start(self):

        #if self._started:
            #msg = 'Client has already been started.'
            #logging.error(msg)
            #raise RuntimeError(msg)

        #logging.info('Establishing connections.')

        ## Connect to server.
        #ctx = self._context
        #server = create_socket('request', context=ctx)
        #server.connect(f'tcp://{self._host}:{Ports.CONTROL}')

        ## Open shell/terminal connection.
        #shell = create_socket('reply', context=ctx)
        #shell.bind(f'tcp://{self._host}:{Ports.CLIENT_SHELL}')

        ## Setup a poller.
        #poller = zmq.Poller()
        #poller.register(server, zmq.POLLIN | zmq.POLLOUT)
        #poller.register(shell, zmq.POLLIN | zmq.POLLOUT)

        ## Store sockets and poller(s).
        #self._sockets['server'] = server
        #self._sockets['shell'] = shell
        #self._poller = poller

        ## Setup namespace within which shell commands will be executed.
        #self._ns = {}



        ## Enter the main loop.
        #self._started = True
        #self.run()


    #@property
    #def host(self) -> str:
        #return self._host

    #@property
    #def status(self):
        #return self.send_get('cam.status')

    #@property
    #def analog_gain(self) -> float:
        #return self.send_get('cam.analog_gain')

    #@property
    #def exposure_mode(self) -> str:
        #return self.send_get('cam.exposure_mode')

    #@exposure_mode.setter
    #def exposure_mode(self, mode: str) -> str:
        #self.send_set('cam.exposure_mode', mode)

    #@property
    #def exposure_speed(self) -> str:
        #return self.send_get('cam.exposure_speed')

    #@property
    #def framerate(self) -> str:
        #return self.send_get('cam.framerate')

    #@framerate.setter
    #def framerate(self, rate: int) -> str:
        #self.send_set('cam.framerate', rate)

    #@property
    #def iso(self) -> str:
        #return self.send_get('cam.iso')

    #@iso.setter
    #def iso(self, val: int) -> str:
        #self.send_set('cam.iso', val)

    #@property
    #def shutter_speed(self) -> str:
        #return self.send_get('cam.shutter_speed')

    #@shutter_speed.setter
    #def shutter_speed(self, speed: int) -> None:
        #self.send_set('cam.shutter_speed', speed)

    #@property
    #def resolution(self) -> Tuple[int, int]:
        #return self.send_get('cam.resolution')

    #@resolution.setter
    #def resolution(self, res: Tuple[int, int]) -> str:
        #self.send_set('cam.resolution', res)

    ##------------------------------------------------------------------------------------#
    ## Main event loop

    #def run(self):
        #"""
        #Event loop that read user input from stdin, interprets them as requests,
        #sends requests, and finally handles replies.

        #The target object (the 'base' of the namespace) is implicitly the client
        #instance itself.


        #"""
        #if not self._started:
            #msg = "Server must be started before being run. Returning."
            #logging.warning(msg)
            #return

        #if self._running:
            #logging.warning('Server is already running. Ignoring call.')
            #return
        #self._running = True

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






    #def close(self):
        #"""Close the client only. Does not close server or other services."""
        #self._terminate = True


    #def _reset_connections(self) -> None:
        #"""
        #Put rep/req sockets into expected modes.

          #- put server socket into POLLOUT
          #- put shell socket into POLLIN

        #"""

        ## Alias
        #server = self._sockets['server']
        #shell = self._sockets['shell']
        #poller = self._poller
        #timeout = 1000

        #raise NotImplementedError

        ## Set server
        ##poll_result = dict(poller.poll(1000))
        ##if server in poll_result:
            ##if can_recv(server, poll_result):
                ##msg = server.recv_json()

        ##if shell in poll_result:
            ##if can_send(shell)


    #def _shutdown(self) -> None:
        #"""Shut everything down neatly."""
        #logging.debug('Closing connections.')
        #for sock in self._sockets.values():
            #sock.close()
        #logging.debug('Terminating threads.')
        #for thread in self._threads.values():
            #thread.close()

        #time.sleep(1.0)
        #self._context.term()
        #self._running = False
        #self._stopped = True
        #logging.info('Client closed.')


    ##------------------------------------------------------------------------------------#
    ## Sending methods


    #def send(self, req: Dict) -> None:
        #"""
        #Final method in pipeline for sending requests to the server.
        #"""
        #if self.verbose:
            #print(f'Sending request: {req}', flush=True)
        #self._prev_request = req
        #self._server_sock.send_json(req)


    #def send_get(self, key: str) -> Any:
        #"""
        #Main gateway for retrieving attributes from the server's side. The server
        #instance is the target (implicitly). Use this method to ensures that that the
        #request is well-formed. Provided as a convenience.

        #Get requests have the following structure:
          #- 'type' : str     Aalways 'get'.
          #- 'key' : str      Name of attribute to set.

        #"""
        #req = {'type' : 'get',
               #'key'  : key}
        #return self.send(req)


    #def send_set(self, key: str, val: Any) -> Any:
        #"""
        #Main gateway for setting attributes on the server side. The server instance
        #is the target (implicitly). Use this method to ensures that that the request
        #is well-formed. Provided as a convenience.

        #Set requests have the following structure:
          #- 'type' : str     Always 'set'.
          #- 'key' : list     Name of attribute to set.
          #- 'val' : dict     Attribute's new value.


        #"""
        #req = {'type' : 'set',
               #'key' : key,
               #'val' : val}
        #return self.send(req)


    #def send_call(self,
                  #key: str,
                  #args: Optional[List] = None,
                  #kw: Optional[Dict] = None,
                  #) -> Any:
        #"""
        #Main gateway for calling methods on the server side. The server instance
        #is the target (implicitly). Use this method to ensures that that the request
        #is well-formed. Provided as a convenience.

        #Call requests have the following structure:
          #- 'type' : str     Always 'call'.
          #- 'key' : str      Name of method to call.
          #- 'args' : list    A possibly empty list of positional arguments.
          #- 'kw' : dict      A possibly empty dictionary of keyword arguments.

        #"""

        #args = [] if args is None else args
        #kw = {} if kw is None else kw
        #req = {'type' : 'call',
               #'key'  : key,
               #'args' : args,
               #'kw' : kw}
        #return self.send(req)


    ##--------------------------------------------------------------------------#
    ## Receiving methods


    #def _handle_reply(self, rep: Dict) -> None:

        #if self.verbose:
            #print(f'Received reply: {rep}', flush=True)

        #rtype = rep.get('type', None)
        #if rtype not in self._reply_handlers.keys():
            #pprint(RuntimeError(f'invalid request type {rtype}.'))

        #self._reply_handlers[rtype](rep)


    #def _handle_return(self, rep: Dict) -> None:
        #"""Handle a reply containing a return value."""
        #val = rep['val']
        #if val is None:
            #return
        #pprint(val)


    #def _handle_error(self, rep: Dict) -> None:
        #"""Handle a reply indicating an exception was raised."""
        #val = rep['val']
        #pprint(val)


    #def _handle_stdin(self, chars: str) -> None:
        #"""
        #Execute code sent entered into stdin. Chars will be prepended with `self`
        #prior to execution.

        #Mimics normal access to the client's attributes and methods.
        #"""
        #chars = 'self.' + chars if not chars.startswith('self.') else chars
        #exec(chars)


    ##------------------------------------------------------------------------------------#
    ## Recording/previewing utilities

    #"""
    #How to have a previewer than uses frames_q?
    #"""



    #def start_recording(self, path, duration) -> None:

        #path = Path(path)
        #if path.exists():
            #print(f'Location: {path} exists. Delete before recording.')
            #return
        #print('Starting recording.')
        #max_FPS = 40
        #max_frames = int(duration * max_FPS)
        #width, height = self.resolution
        #shape = (max_frames, height, width)

        #self.clear_frame_receiver()
        #self.recording = True
        #self.t_start = clock()

        #self.frame_receiver = H5Receiver(self.frame_sock,
                                         #path,
                                         #shape,
                                         #dtype=np.uint8)
        #self.frame_receiver.start()
        #return self.call('start_recording', args=[duration], target='server')


    #def check_recording(self):

        #if not self.recording:
            #print('Not recording.')
            #return

        #if self.frame_receiver.complete:
            #print('H5 receiver full. Stopping recording.')
            #self.stop_recording()


    #def stop_recording(self) -> None:

        #if not self.recording:
            #print('Not recording')
        #ret = self.call('stop_recording', target='server')
        #self.recording = False
        #t_stop = clock()
        #elapsed = t_stop - self.t_start
        #n_frames = self.frame_receiver.n_received

        #with self.frame_receiver.lock:
            #try:
                #self.frame_receiver.file.close()
                #self.frame_receiver.terminate = True
            #except:
                #pass


        #FPS = n_frames / elapsed
        #print(f'Received {n_frames} frames in {elapsed:.2f}'
              #f'secs. (FPS={FPS:.2f})')
        #return ret



