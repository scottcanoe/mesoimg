#!/usr/bin/python
import argparse
import os
import subprocess as sp
import sys
from mesoimg.osutils import *
from mesoimg.app import *
from mesoimg.server import MesoServer



server = MesoServer()
server.run()
