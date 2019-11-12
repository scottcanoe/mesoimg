import logging
from typing import Any, Callable, Optional
import time as _time
import numpy as np


__all__ = [
    'time',
    'today',
    'Clock',
    'Timer',
]


def time() -> float:
    """
    Get current UNIX time. Wraps built-in ``time.time`` to keep all time
    related functions in the same place to simplify imports.
    """
    return _time.time()


sleep = _time.sleep


def today() -> str:
    """
    Returns the ISO-formatted local date (e.g., 2019-11-08).
    """
    d = time.localtime()
    return f'{d.tm_year}-{d.tm_mon:02}-{d.tm_mday:02}'



class Clock:

    """
    Class to assist in timing intervals.
    """
    
    
    def __init__(self,
                 *,
                 start: bool = True,
                 time_fn: Callable[[], float] = _time.perf_counter):

        self._time_fn = time_fn
        self.reset()
        if start:
            self.start()


    @property
    def running(self) -> bool:
        return self._running
        

    def reset(self) -> None:
        self._t_start = None
        self._t_stop = None
        self._running = False
        
        
    def restart(self) -> float:
        self.check_running()
        self.reset()
        return self.start()
        

    def start(self) -> float:
        """
        Start the clock. Returns 0.
        """
        self.check_not_running()
        self._t_start = self._time_fn()
        self._running = True
        return 0
        
        
    def time(self) -> float:
        """
        Get the clock's current time.
        """
        self.check_running()
        t = self._time_fn() - self._t_start
        return t


    def stop(self) -> float:
        """
        Stop the clock.
        """
        self._t_stop = self.time()
        self._running = False
        return self._t_stop

        

    def check_running(self) -> None:
        if not self._running:
            raise RuntimeError('Clock is not running.')


    def check_not_running(self) -> None:
        if self._running:
            raise RuntimeError('Clock is running.')
        
    

class Timer:

    """
    Class to assist in timing intervals.
    """
    
    def __init__(self,
                 name: str = '',
                 verbose: bool = False,
                 time_fn: Callable[[], float] = _time.perf_counter,
                 logger: Optional[logging.Logger] = None):

        self._name = name
        self._verbose = verbose
        self._time_fn = time_fn
        self._logger = logger
        self.reset()

    @property
    def name(self) -> str:
        return self._name
        
    @property
    def running(self) -> bool:
        return self._running

    @property
    def timestamps(self) -> np.ndarray:
        if isinstance(self._timestamps, list):
            return np.array(self._timestamps)
        return self._timestamps

        
    def reset(self) -> 'Timer':
        self._t_start = None
        self._t_stop = None
        self._running = False
        self._timestamps = []
                        
                    
    def start(self) -> 'Timer':
        self.check_not_running()
        self._t_start = self._time_fn()
        self._timestamps = [0]
        self._running = True
        return self
            
    def tic(self) -> 'Timer':
        self.check_running()
        t = self._time_fn() - self._t_start
        self._timestamps.append(t)
        return self

    def stop(self) -> 'Timer':
        self.check_running()
        self._t_stop = self._time_fn()
        self._running = False
        self._timestamps = np.array(self._timestamps)
        if self._verbose:
            self.print_summary()


    def check_running(self) -> None:
        if not self._running:
            raise RuntimeError('Timer is not running.')


    def check_not_running(self) -> None:    
        if self._running:
            raise RuntimeError('Timer is running.')


    def print_summary(self) -> None:
        s = "<Timer "
        if self.name:
            s = f"Timer('{self.name}') finished: "
        else:            
            s = "<Timer finished: "        
        s += f'time={self._t_stop} sec.>'
        print(s, flush=True)
        
