#!/usr/bin/python
import argparse
import os
import subprocess as sp
import sys
from mesoimg.osutils import *
from mesoimg.app import *
from mesoimg.server import MesoServer


"""
Can either start the server in blocking mode or open a command line interface.
"""

server = MesoServer()
server.start()
