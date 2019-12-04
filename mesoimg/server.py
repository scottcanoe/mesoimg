from queue import Queue
from threading import Event, Lock, Thread
import time
from time import perf_counter as clock
from typing import Dict
import numpy as np
import zmq
from mesoimg import *



class MesoServer:

    def __init__(self):        
        self.context = zmq.Context()
        self.cmd_sock = self.context.socket(zmq.REP)
        self.cmd_sock.bind(f'tcp://*:{Ports.COMMAND}')

        self.open_cam()
        self.cam_thread = None


    def open_sockets(self):
        print('Opening sockets.', flush=True)
        self.context = zmq.Context()
        self.cmd_sock = self.context.socket(zmq.REP)
        self.cmd_sock.bind(f'tcp://*:{COMMAND_PORT}')
        self.frame_sock = self.context.socket(zmq.PUSH)
        self.frame_sock.bind(f'tcp://*:{FRAME_PORT}')
        self.status_sock = self.context.socket(zmq.PUSH)
        self.status_sock.bind(f'tcp://*:{STATUS_PORT}')
    
    def close_sockets(self):
        self.cmd_sock.close()
        self.frame_sock.close()
        self.status_sock.close()
        self.context.term()
            
    def open_cam(self):
        self.cam = Camera(frame_sock=self.frame_sock,
                          status_sock=self.status_sock)
            

    def close_cam(self):
        self.cam.close()

            
            
    def start(self):
        """
        
        get:  {'target' : 'cam',
               'action' : 'get',
               'key'    : 'exposure_speed'}

        set:  {'target' : 'cam',
               'action' : 'set',               
               'key'    : 'exposure_speed',
               'val'    : 'off'}

        call: {'target' : 'cam',
               'action' : 'call',                
               'fn'     : 'start',
               'args'   : [5.0],
               'kw'     : {})}

        
        """
        
        self.terminate = False
        
        print('Ready for commands.', flush=True)
        while not self.terminate:

            cmd = self.cmd_sock.recv_json()
            print(f'Received command = {cmd}', flush=True)
            action = cmd['action']
            target = cmd['target']
            if target == 'cam':
                target = self.cam
            elif target == 'server':
                target = self
            else:
                msg = f'Unrecognized target {target}.'
                print(msg)
                self.cmd_sock.send_json({'return' : msg})
                continue

            # Handle shortcuts.
            if action == 'get':
                val = getattr(target, cmd['key'])
                time.sleep(0.01)
                self.cmd_sock.send_json({'return' : val})
            
            elif action == 'set':
                setattr(target, cmd['key'], cmd['val'])
                time.sleep(0.01)
                self.cmd_sock.send_json({'return' : 0})
                
            elif action == 'call':
                fn = getattr(target, cmd['fn'])
                args = cmd['args']
                kw = cmd['kw']
                val = fn(*args, **kw)
                time.sleep(0.01)
                self.cmd_sock.send_json({'return' : val})
            
            else:
                msg = f'Unrecognized action {action}'
                print(msg)
                self.cmd_sock.send_json({'return' : msg})


            if action == 'start_preview':
                self.cam_thread = Thread(target=self.cam.start_preview)
                self.cam_thread.start()
                self.return_()
                continue

            if action == 'stop_preview':
                self.cam._stop = True
                self.return_()
                continue
            
            if action == 'quit':
                self.return_()
                break
                
            # Select the target (server or camera)
            target = cmd.get('target', 'cam')
            if target == 'server':
                target = self
            elif target == 'cam':
                target = self.cam
            else:
                print(f'Unrecognized target {target}')
                self.return_()
                continue
                
            # Select the action ('call' or 'set')
            if action not in ('get', 'set', 'call'):
                print(f'Unrecognized action {action}')
                self.return_(f'Unrecogonized action {action}')
                continue
                                                
        
        
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
    
    def quit(self):
        self.terminate = True
        self.cmd_sock.close()
        self.frame_sock.close()
        self.status_sock.close()
        self.context.term()
        return 0
                    
server = MesoServer()
server.start()


















