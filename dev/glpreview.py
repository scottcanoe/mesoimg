from threading import Lock, Thread
import time
import queue
from queue import Queue
import numpy as np
import zmq
from mesoimg import *
from glumpy import app, gloo, gl, glm


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


def callback(data: np.ndarray) -> None:
    """
    Doctring for f1
    """
    try:
        q.put(data, block=False)
    except queue.Full:
        q.get(False)
        q.put(data, block=False)


lock = Lock()
width, height = 1640, 1232
frame_rgb = np.zeros([height, width, 3], dtype=np.uint8)
frame = frame_rgb[:, :, 0]

FRAME_PUB = 7011
q = Queue(maxsize=30)

sub = Subscriber(recv_frame)
sub.connect(f'tcp://pi-meso.local:{FRAME_PUB}')
sub.subscribe(b'')
sub.callback = callback
sub.start()

win = app.Window(width=width, height=height, aspect=1, vsync=True)

@win.event
def on_draw(dt):
    global q, frame, frame_rgb
    with lock:

        if q.qsize() > 1:
            frame = q.get(block=False)
        if frame.ndim == 2:
            frame_rgb[:, :, 0] = frame[:]
            frame_rgb[:, :, 1] = frame[:]
            frame_rgb[:, :, 2] = frame[:]
        elif frame.ndim == 3:
            frame_rgb[:] = frame[:]

    win.clear()
    quad['texture'] = frame_rgb
    quad.draw(gl.GL_TRIANGLE_STRIP)

quad = gloo.Program(vertex, fragment, count=4)
quad['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
quad['texcoord'] = [( 0, 1), ( 0, 0), ( 1, 1), ( 1, 0)]
quad['texture'] = frame_rgb
app.run()

#sub.stop()
