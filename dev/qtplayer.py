from collections import deque
from pathlib import Path
import sys
from threading import Lock, Thread
import time
from typing import (Optional, Tuple)
import queue

import matplotlib.colors as mpc
from matplotlib import cm
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
import zmq

from mesoimg import *


pg.setConfigOptions(imageAxisOrder='row-major')
QSizePolicy = QtGui.QSizePolicy
Policy = QtGui.QSizePolicy.Policy


class CameraPreview(QtGui.QMainWindow):

    """
    Widget that plays movies.

    The viewing area is a pyqtgraph ImageItem in a GraphicsLayout
    viewbox.

    """

    frame_sub: zmq.Socket
    frame_times: deque
    frame: Optional[Frame] = None

    init_width: int = 800
    init_height: int = 600


    def __init__(self):
        super().__init__()

        self.ctx = zmq.Context()
        self.frame_sub = self.ctx.socket(zmq.SUB)
        self.frame_sub.connect(f'tcp://pi-meso.local:{Ports.CAM_FRAME}')
        self.frame_sub.subscribe(b'')
        self.frame_sub.rcvtimeo = 10

        self.frame_times = deque(maxlen=100)
        self.frame = None

        self.setGeometry(50, 50, self.init_width, self.init_height)
        self.setWindowTitle('Preview')

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

        # - Add a image item to the view box.
        self._image = pg.ImageItem()
        self._view_box.addItem(self._image)


        #-----------------------------------------------------------------------
        # Initialize histogram


        # - Initialize other attributes and state variables.

        self.hist = pg.HistogramLUTItem(self._image)
        self._graphics_layout.addItem(self.hist)


        self.input_timer = QtCore.QTimer()
        self.input_timer.timeout.connect(self.get_input)
        self.input_timer.start(20)

        self._update()
        self.show()


    @property
    def levels(self):
        return self._image.getLevels()


    #--------------------------------------------------------------------------#
    # Public methods


    def get_input(self) -> None:
        """
        QTimer calls this every 20 msec or so to try and grab
        new frame data.
        """
        try:

            topic = self.frame_sub.recv()
            fm = recv_frame(self.frame_sub)
        except zmq.error.Again:
            return

        self.frame = fm
        self._update()


    #--------------------------------------------------------------------------#
    # Private methods


    def _update(self):
        self._update_image()
        fm = self.frame
        if fm:
            ix, t = fm.index, fm.time
            self.frame_times.append(t)
            if ix % 30 == 0:
                tarr = np.array(list(self.frame_times))
                diffs = np.ediff1d(tarr)
                fps = 1 / np.mean(diffs)
                msg = "index={}, time={:.2f}, fps={:.2f}".format(ix, t, fps)
                print(msg)



    def _update_image(self):

        """Gets called by cur_frame.setter(). Bounds are guaranteed to be good.
        """

        if self.frame is None:
            # No data: black screen, empty label.
            self._image.setImage(np.zeros((8, 8)), levels=(0, 1))
            return

        data = self.frame.data
        data = data.T
        data = np.flipud(data)

        self._image.setImage(data)


    #--------------------------------------------------------------------------#
    # Reimplemented methods


    def keyPressEvent(self, event):

        key = event.key()
        mods = QtGui.QApplication.queryKeyboardModifiers()

        # Quit/close.
        if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_Q) and \
           mods == QtCore.Qt.ControlModifier:
            self.close()




app = QtWidgets.QApplication(sys.argv)
win = CameraPreview()

app.exec_()
time.sleep(0.5)
win.frame_sub.close()

