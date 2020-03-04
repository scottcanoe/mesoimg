import itertools
import threading
from threading import Condition, Event, Lock, RLock, Thread
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import queue
import numpy as np
import zmq
#from mesoimg.app import Ports
#from mesoimg.common import *
#from mesoimg.inputs import *
#from mesoimg.messaging import *

from pygments.lexers.python import PythonLexer
from pygments.styles import get_style_by_name
from prompt_toolkit import prompt
import prompt_toolkit as ptk
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles.pygments import style_from_pygments_cls





class MesoConsole:

    completer = WordCompleter(['cam', 'server', 'status'])

    lexer = PygmentsLexer(PythonLexer)

    style = style_from_pygments_cls(get_style_by_name('monokai'))


    def __init__(self):


        # Initialize the prompt toolkit.
        self.pses = ptk.PromptSession('> ',
                                      completer=self.completer,
                                      #lexer=self.lexer,
                                      include_default_pygments_style=False,
                                      style=self.style)


        self.terminate = Event()
        self.event_loop_finished = Event()


    def start(self):
        # Printer a header.
        print('\n')
        print('            MesoConsole')
        print('------------------------------------\n')

        self.run()


    def run(self):

        while not self.terminate.is_set():

            try:
                s = self.prompt()
                if s == 'close()':
                    self.terminate.set()
                    continue

            except KeyboardInterrupt:
                self.terminate.set()
                continue

        self.event_loop_finished.set()



    def prompt(self) -> str:
        return self.pses.prompt()


    def close(self) -> None:
        self.terminate.set()
        self.event_loop_finished.wait()
        print('Exiting...\n')



#console = MesoConsole()
#console.start()
#p = ptk.PromptSession()

class Prompt(Thread):

    def __init__(self):
        super().__init__()
        self.terminate = Event()
        self.start()


    def run(self):
        while not self.terminate.is_set():
            s = prompt('> ')
            if s == 'q':
                break


print('Started.', flush=True)

#p = Prompt()