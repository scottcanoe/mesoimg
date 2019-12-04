from pathlib import Path
from threading import Lock, Thread
import time
from time import perf_counter as clock
from typing import Optional, Tuple, Union
import h5py
import imageio
import numpy as np
import matplotlib.pyplot as plt
import zmq
from mesoimg.common import *


HOST = 'pi-meso.local'
COMMAND_PORT = 7000
FRAME_PORT = 7001
STATUS_PORT = 7002
PREVIEW_PORT = 7010



frame = None
n_received = 0
status = None
msg = None



class FrameReceiver(Thread):
    
    """
    Receives frames from frame socket. Only stores 1-most recent.
    
    """
    
    def __init__(self, sock):
        super().__init__()
        self.sock = sock
        self.lock = Lock()
        self.data = None
        self.n_received = 0
        self.terminate = False

    
    def run(self):

        global frame
        global frames_received

        while not self.terminate:
            #self.sock.send_string('ready')
            fm = recv_frame(self.sock)
            with self.lock:
                self.frame = fm
                self.n_received += 1
                frame = fm
                frames_received += 1



class PreviewReceiver(Thread):
    
    """
    Receives frames from frame socket. Only stores 1-most recent.
    
    """
    
    def __init__(self, frame_sock, preview_sock):
        super().__init__()
        self.frame_sock = frame_sock
        self.preview_sock = preview_sock
        self.lock = Lock()
        self.frame = None
        self.n_received = 0
        self.t_last = None
        self.terminate = False

        
    def run(self):
        
        global frame
        global n_received

        frame = None
        n_received = 0
        
        t_last = 0
        while not self.terminate:

            fm = recv_frame(self.frame_sock)
            
            cur_t = clock()
            with self.lock:
                self.frame = fm
                frame = fm
                self.n_received += 1
                if cur_t - t_last > 0.5:
                    send_frame(self.preview_sock, self.frame)
                    t_last = cur_t



class H5Receiver(Thread):
    
    """
    Receives frames from frame socket. Only stores 1-most recent.
    
    """
    
    def __init__(self, frame_sock, path, shape, dtype=np.uint8):
        super().__init__()
        self.frame_sock = frame_sock
        self.lock = Lock()
        self.frame = None
        self.n_received = 0
        self.t_last = None
        self.terminate = False
        
        self.path = Path(path)
        self.file = h5py.File(str(self.path), 'w')
        
        self.max_frames = shape[0]
        self.dset = self.file.create_dataset('data', shape, dtype=dtype)
        self.dset.attrs['index'] = 0
        self.ts = self.file.create_dataset('timestamps', (shape[0],), dtype=float)
        
        self.complete = False
        self.terminate = False
        
        


    
    def run(self):
        
        global frame
        global n_received
        
        frame = None
        n_received = 0
        
        while not self.terminate:

            fm = recv_frame(self.frame_sock)
            frame = fm
            n_received += 1
            
            with self.lock:
                if not self.complete:                    
                    index = self.dset.attrs['index']
                    self.dset[index, ...] = fm.data
                    self.ts[index] = fm.timestamp
                    self.dset.attrs['index'] += 1
                    if self.dset.attrs['index'] >= self.max_frames:
                        self.complete = True
                        self.file.close()
                        self.terminate = True
                    self.frame = fm
                    self.n_received += 1
                    
    
class StatusReceiver(Thread):
    
    """
    Receives status dictionaries from status socket. Only stores 1-most recent.
    
    """
    
    def __init__(self, sock):
        super().__init__()
        self.sock = sock
        self.lock = Lock()
        self.data = None
        self.n_received = 0
        self.terminate = False

    
    def run(self):

        global status
        while not self.terminate:
            #self.sock.send_string('ready')
            stat = self.sock.recv_json()
            with self.lock:
                status = stat
                self.data = stat
                self.n_received += 1
        



class MesoClient:
    
    def __init__(self):

        print('Starting client.', flush=True)

        self.context = zmq.Context()        

        self.cmd_sock = self.context.socket(zmq.REQ)
        self.cmd_sock.connect(f'tcp://{HOST}:{COMMAND_PORT}') 

        self.frame_sock = self.context.socket(zmq.PULL)
        self.frame_sock.connect(f'tcp://{HOST}:{FRAME_PORT}')    

        self.status_sock = self.context.socket(zmq.PULL)
        self.status_sock.connect(f'tcp://{HOST}:{STATUS_PORT}')

        self.preview_sock = self.context.socket(zmq.PUSH)
        #self.preview_sock.bind(f'tcp://127.0.0.1:{PREVIEW_PORT}')
        self.preview_sock.bind(f'tcp://127.0.0.1:{PREVIEW_PORT}')
        
        self.frame_receiver = None
        self.status_receiver = StatusReceiver(self.status_sock)
        
        self.previewing = False
        self.recording = False
    
        time.sleep(1)
        
    @property
    def status(self):
        return self.get('status')

    @property
    def analog_gain(self) -> float:
        return self.get('analog_gain')

    @property
    def exposure_mode(self) -> str:
        return self.get('exposure_mode')
    
    @exposure_mode.setter
    def exposure_mode(self, mode: str) -> str:
        self.set('exposure_mode', mode)

    @property
    def exposure_speed(self) -> str:
        return self.get('exposure_speed')
    
    @exposure_speed.setter
    def exposure_speed(self, speed: int) -> None:
        print('cannot directly set exposure speed.')
        #self.set('exposure_speed', speed)
    
    @property
    def framerate(self) -> str:
        return self.get('framerate')
    
    @framerate.setter
    def framerate(self, rate: int) -> str:
        self.set('framerate', rate)
    
    @property
    def iso(self) -> str:
        return self.get('iso')
    
    @iso.setter
    def iso(self, val: int) -> str:
        self.set('iso', val)
        
    @property
    def shutter_speed(self) -> str:
        return self.get('shutter_speed')
    
    @shutter_speed.setter
    def shutter_speed(self, speed: int) -> None:
        self.set('shutter_speed', speed)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.get('resolution')
        
    @resolution.setter
    def resolution(self, res: Tuple[int, int]) -> str:
        self.set('resolution', res)


    def get(self, key, target='cam'):
        cmd = {'target' : target,
               'action' : 'get',
               'key' : key}
        print(f'Sending command: {cmd}')
        self.cmd_sock.send_json(cmd)
        return self.cmd_sock.recv_json()['return']
    
    
    def set(self, key, val, target='cam'):
        cmd = {'target' : target,
               'action' : 'set',
               'key' : key,
               'val' : val}
        print(f'Sending command: {cmd}')
        self.cmd_sock.send_json(cmd)
        return self.cmd_sock.recv_json()['return']
    
    
    def call(self, fn, args=None, kw=None, target='cam'):
        args = [] if args is None else args
        kw = {} if kw is None else kw
        cmd = {'target' : target,
               'action' : 'call',
               'fn' : fn,
               'args' : args,
               'kw' : kw}
        print(f'Sending command: {cmd}')
        self.cmd_sock.send_json(cmd)
        return self.cmd_sock.recv_json()['return']
    
        
    
    def start_preview(self) -> None:
        
        self.clear_frame_receiver()
        self.previewing = True
        self.t_start = clock()
        self.frame_receiver = PreviewReceiver(self.frame_sock, self.preview_sock)        
        self.frame_receiver.start()
        return self.call('start_preview', target='server')
        

    def stop_preview(self) -> None:

        if not self.previewing:
            print('Not previewing')
        ret = self.call('stop_preview', target='server')
        self.previewing = False
        t_stop = clock()
        elapsed = t_stop - self.t_start
        n_frames = self.frame_receiver.n_received
        FPS = n_frames / elapsed
        print(f'Received {n_frames} frames in {elapsed:.2f} secs. (FPS={FPS:.2f})')
        return ret


    def start_recording(self, path, duration) -> None:
        
        path = Path(path)        
        if path.exists():
            print(f'Location: {path} exists. Delete before recording.')
            return
        print('Starting recording.')
        max_FPS = 40
        max_frames = int(duration * max_FPS)
        width, height = self.resolution
        shape = (max_frames, height, width)
        
        self.clear_frame_receiver()
        self.recording = True
        self.t_start = clock()

        self.frame_receiver = H5Receiver(self.frame_sock,
                                         path,
                                         shape,
                                         dtype=np.uint8)
        self.frame_receiver.start()
        return self.call('start_recording', args=[duration], target='server')
        
        
    def check_recording(self):

        if not self.recording:
            print('Not recording.')
            return
            
        if self.frame_receiver.complete:
            print('H5 receiver full. Stopping recording.')
            self.stop_recording()


    def stop_recording(self) -> None:

        if not self.recording:
            print('Not recording')
        ret = self.call('stop_recording', target='server')
        self.recording = False
        t_stop = clock()
        elapsed = t_stop - self.t_start
        n_frames = self.frame_receiver.n_received
        
        with self.frame_receiver.lock:
            try:                
                self.frame_receiver.file.close()
                self.frame_receiver.terminate = True
            except:
                pass
            
                
        FPS = n_frames / elapsed
        print(f'Received {n_frames} frames in {elapsed:.2f} secs. (FPS={FPS:.2f})')
        return ret
    
    
    def clear_frame_receiver(self) -> None:
        
        if self.frame_receiver:
            self.frame_receiver.terminate = True
            self.frame_receiver = None
    
    
    def clear_status_receiver(self) -> None:
        
        if self.status_receiver:
            self.status_receiver.terminate = True
            self.status_receiver = None
            
    def quit(self) -> None:
        cmd = {'action' : 'call',
               'fn' : 'quit',
               'target' : 'server'}
        self.cmd_sock.send_json(cmd)
        msg = self.cmd_sock.recv_json()['return']
        self.cmd_sock.close()
        self.frame_sock.close()
        self.status_sock.close()
        self.context.term()
        return msg
        

def record(client, path, duration):

    path = Path(path)
    if path.exists():
        path.unlink()
    
    client.start_recording(path, duration)
    time.sleep(duration)
    client.stop_recording()
    
    f = h5py.File(str(path), 'r')
    dset = f['data']
    timestamps = f['timestamps']
    n_frames = dset.attrs['index']
    mov = dset[0:n_frames, :, :]
    ts = timestamps[0:n_frames]
    f.close()    
    
    mp4_path = path.parent / 'mov.mp4'
    if mp4_path.exists():
        mp4_path.unlink()
    imageio.mimwrite(mp4_path, mov, fps=40)


import time
c = MesoClient()
c.exposure_mode = 'verylong'
c.iso = 800

c.start_preview()
#record(c, '/Users/scott/test.h5', 0.5 * 60)

#path = Path('/Users/scott/test.h5')
#f = h5py.File(str(path), 'r')
#dset = f['data']
#timestamps = f['timestamps']
#n_frames = dset.attrs['index']
#mov = dset[0:n_frames, :, :]
#ts = timestamps[0:n_frames]
#f.close()

#mov = (255 * (mov / 10)).astype(np.uint8)
#mov[mov > 255] = 255
#save_path = path.parent / 'test.mp4'
#if save_path.exists():
    #save_path.unlink()
#imageio.mimwrite(save_path, mov, fps=40)