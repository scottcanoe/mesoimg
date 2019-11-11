import io
import os
from pathlib import Path
import socket
import time
from typing import (Any,
                    Callable,
                    Iterable,
                    List,
                    Mapping,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)
from urllib.parse import urlparse, ParseResult
import h5py
import numpy as np
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
from mesoimg.common import *






class H5Interface:


    def __init__(self, parent: 'Camera', url: URLLike):
        self.parent = parent
        self.url = parse_url(url)
        self.file = None
        
        
    def open(self):
        self.file = h5py.File(
