from pathlib import Path
import sys
from typing import Any, Optional
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
pg.setConfigOptions(imageAxisOrder='row-major')
from mesoimg import *


class ROIMaker(QtGui.QMainWindow):

    """
    Widget that plays movies.

    The viewing area is a pyqtgraph ImageItem in a GraphicsLayout
    viewbox.

    """

    init_width = 800
    init_height = 800
    _mov = None


    def __init__(self,
                 im: np.array,
                 parent: Optional[Any] = None,
                 ):
        super().__init__()

        self.im = im

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, self.init_width, self.init_height)
        self.setWindowTitle("ROI Maker")

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
        self._layout.setStretch(0, 800)

        # - Add a view box to enable scaling/panning and mouse drags.
        self._view_box = self._graphics_layout.addViewBox(lockAspect=True,
                                                          row=0, col=0,
                                                          invertY=True)
        self._view_box.setMenuEnabled(False)

        # - Add a image item to the view box.
        self._image = pg.ImageItem()
        self._view_box.addItem(self._image)
        self._image.setImage(self.im)

        self.roi = pg.PolyLineROI([[10, 10], [10, 30], [30, 30], [30, 10]],
                                  closed=True)
        self._view_box.addItem(self.roi)



        self.show()


# create GUI

#w = pg.GraphicsWindow(size=(1000, 800), border=True)
#w.setWindowTitle('ROI Maker')

#w2 = w.addLayout(row=0, col=0)

#v2a = w2.addViewBox(row=1, col=0, lockAspect=True)
#r2a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
#v2a.addItem(r2a)
#r2b = pg.PolyLineROI([[0,-20], [10,-10], [10,-30]], closed=False)
#v2a.addItem(r2b)
#v2a.disableAutoRange('xy')

#v2a.autoRange()

import matplotlib.pyplot as plt
from pyqtgraph import ROI



inpath = Path.home() / "mov.h5"
mov = read_h5(inpath)
im = np.max(mov, axis=0)


app = QtGui.QApplication([])
win = ROIMaker(im)
app.exec()

roi = win.roi

# Get the region's bounding rectangular region with non-roi areas
# zerod out.
box = roi.getArrayRegion(im, win._image)

# Get the row/column slices for this box.
rows, cols = roi.getArraySlice(im, win._image)[0]


plt.imshow(im[rows, cols])
plt.show()