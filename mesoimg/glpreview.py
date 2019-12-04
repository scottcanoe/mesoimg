from threading import Lock, Thread
import time
from glumpy import app, gloo, gl, glm
import numpy as np
import zmq
from mesoimg.common import *






class FrameReceiver(Thread):
    
    """
    Receives frames from frame socket. Only stores 1-most recent.
    
    """
    
    def __init__(self, sock):
        super().__init__()
        self.sock = sock
        self.data = None
        self.n_received = 0
        self.terminate = False

    
    def run(self):

        global frame, frame_base, n_received

        while True:
            with lock:
                if self.terminate:
                    break

            fm = recv_frame(self.sock)
            with lock:
                frame_base = fm
                data = fm.data
                pmin, pmax = np.percentile(data, [0.5, 99.5])
                print(f'99.5 percentile: {pmax}')
                frame[:, :, 0] = data
                frame[:, :, 1] = data
                frame[:, :, 2] = data
                n_received += 1
                self.data = data
                self.n_received += 1

                

lock = Lock()
frame_base = np.zeros([480, 640], dtype=np.uint8)
frame = np.zeros([480, 640, 3], dtype=np.uint8)
n_received = 0
                
                
vertex = """
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
"""

fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    }
"""


context = zmq.Context()
sock = context.socket(zmq.PULL)
sock.connect(f'tcp://127.0.0.1:{PREVIEW_PORT}')

frame_receiver = FrameReceiver(sock)
frame_receiver.start()

window = app.Window(width=640, height=480, aspect=1)

@window.event
def on_draw(dt):
    window.clear()
    quad['texture'] = frame
    quad.draw(gl.GL_TRIANGLE_STRIP)

quad = gloo.Program(vertex, fragment, count=4)
quad['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
quad['texcoord'] = [( 0, 1), ( 0, 0), ( 1, 1), ( 1, 0)]
quad['texture'] = frame
app.run()

frame_receiver.terminate = True
time.sleep(0.5)
sock.close()
context.term()
