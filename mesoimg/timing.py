from typing import Any, Callable, Optional
import time as _time
import numpy as np



def time() -> float:
    """
    Wrapper for built-in ``time.time`` to keep all time
    related functions in the same place to simplify imports.
    """
    return _time.time()


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
                 *,
                 start: bool = True,
                 time_fn: Callable[[], float] = _time.perf_counter,
                 verbose: bool = False,
                 logger: Optional[logging.Logger] = None):

        self._name = name
        self._time_fn = time_fn
        self._verbose = verbose
        self._logger = logger
        self.reset()
        if start:
            self.start()

    @property
    def name(self) -> str:
        return self._ID
        
    @property
    def running(self) -> bool:
        return self._running

    @property
    def timestamps(self) -> np.ndarray:        
        return np.array(self._timestamps)
        
        
    def reset(self):
        self._t_start = None
        self._t_stop = None
        self._running = False
        self._timestamps = []
                        
                    
    def start(self) -> float:
        self.check_not_running()
        self._t_start = self._time_fn()
        self._timestamps = [0]
        self._running = True
        return 0
        
        
    def tic(self) -> float:
        self.check_running()
        t = self._time_fn() - self._t_start
        self._timestamps.append(t)
        return t


    def check_running(self) -> None:
        if not self._running:
            raise RuntimeError('Timer is not running.')


    def check_not_running(self) -> None:    
        if self._running:
            raise RuntimeError('Timer is running.')


    def print_summary(self) -> None:
        s = '<Timer ('
        if self._name:
            s += f'name={self._name}, '
        s += f'elapsed={self._t_stop}>'
        print(s, flush=True)
        
