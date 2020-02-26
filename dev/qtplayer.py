from collections import deque
from threading import Lock, Thread
import time
import queue
from queue import Queue
import numpy as np
import zmq
from mesoimg import *

from pathlib import Path
import sys
import matplotlib.colors as mpc
from matplotlib import cm
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
import pyqtgraph as pg

QSizePolicy = QtGui.QSizePolicy
Policy = QtGui.QSizePolicy.Policy


class MoviePlayer(QtGui.QMainWindow):

    """
    Widget that plays movies.

    The viewing area is a pyqtgraph ImageItem in a GraphicsLayout
    viewbox.

    """

    init_width = 800
    init_height = 600

    def __init__(self, frame_q: Queue,
                       cmap: str = 'gray',
                       ):

        super().__init__()

        self.frame_q = frame_q
        self._cmap = cmap
        self.frame = None

        pg.setConfigOptions(imageAxisOrder='row-major')

        self.setGeometry(50, 50, self.init_width, self.init_height)
        self.setWindowTitle('Movie Player')

        # Initialize a central widget and a main layout.
        self._central_widget = QtGui.QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._layout = QtGui.QVBoxLayout()
        self._central_widget.setLayout(self._layout)
        self.setAutoFillBackground(True)

        #-----------------------------------------------------------------------
        # Initialize image viewing area.


        # - Initialize a graphics layout for the image area.
        self._graphics_layout = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self._graphics_layout)

        # - Add a view box to enable scaling/panning and mouse drags.
        self._view_box = self._graphics_layout.addViewBox(lockAspect=True,
                                                          row=0,
                                                          col=0,
                                                          invertY=True)
        self._view_box.setMenuEnabled(False)
        #self._view_box.setSizePolicy(QSizePolicy(Policy(1), Policy(5)))

        # - Add a image item to the view box.
        self._image = pg.ImageItem()
        self._view_box.addItem(self._image)

        #-----------------------------------------------------------------------
        # Initialize histogram

        #self._hist = pg.PlotWidget()
        #hist = Histogram()
        #self._hist = hist
        #self._layout.addWidget(self._hist)

        # - Initialize other attributes and state variables.

        self.hist = pg.HistogramLUTItem(self._image)
        self._graphics_layout.addItem(self.hist)
        #self.hist.setLevels(0, 255)
        #self._layout.addWidget(self.hist)

        self._update()
        self.show()


    @property
    def levels(self):
        return self._image.getLevels()


    #--------------------------------------------------------------------------#
    # Public methods



    #--------------------------------------------------------------------------#
    # Private methods

    def _update(self):
        self._update_image()
        self._update_histogram()

    def _update_image(self):

        """Gets called by cur_frame.setter(). Bounds are guaranteed to be good.
        """

        if self.frame is None:
            # No data: black screen, empty label.
            self._image.setImage(np.zeros((8, 8)), levels=(0, 1))
            return

        fm = self.frame[:]
        #fm = self._smap.to_rgba(self._mov[i_frame])

        self._image.setImage(fm)


    def _update_histogram(self):
        pass


    #--------------------------------------------------------------------------#
    # Reimplemented methods

    def keyPressEvent(self, event):

        key = event.key()
        mods = QtGui.QApplication.queryKeyboardModifiers()

        # Quit/close.
        if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_Q) and \
           mods == QtCore.Qt.ControlModifier:
            self.close()


class Histogram(pg.PlotWidget):

    def __init__(self, parent=None, background='default', **kargs):
        super().__init__(parent=parent, background=background, **kargs)

    def sizeHint(self):
        return (100, 100)

    def sizePolicy(self):
        return QSizePolicy(Policy(0), Policy(0))


timestamps = deque(maxlen=100)

def callback(frame: np.ndarray) -> None:
    """
    Doctring for f1
    """
    #print(f'callback: {frame}')
    try:
        frame_q.put(frame, block=False)
    except queue.Full:
        frame_q.get(False)
        frame_q.put(frame, block=False)

    win.frame = frame
    win._update_image()
    try:

        ix, ts = frame.index, frame.timestamp
        timestamps.append(ts)
        if ix % 30 == 0:
            ts = np.array(list(timestamps))
            diffs = np.ediff1d(ts)
            fps = 1 / np.mean(diffs)
            print(f'fps: {fps}')
            print(f'{frame}')
    except Exception as exc:
        print(exc)


from mesoimg import Ports

frame_q = Queue(maxsize=30)

app = QtWidgets.QApplication(sys.argv)
win = MoviePlayer(frame_q)


sub = Subscriber(recv_frame)
sub.connect(f'tcp://pi-meso.local:{Ports.CAM_FRAME}')
sub.subscribe(b'')
sub.callback = callback
sub.start()

app.exec_()
sub.stop()
time.sleep(0.5)
sys.exit(0)

