from collections import OrderedDict
import logging
from typing import Any, Callable, Optional
import time
import numpy as np


__all__ = [
    'Clock',
    'Timer',
]




class Clock:

    """
    Class to assist in timing intervals.
    """
    
    
    def __init__(self,
                 *,
                 start: bool = True,
                 time_fn: Callable[[], float] = time.perf_counter):

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
    Class for ticking off intervals.
    """
    
    def __init__(self,
                 name: str = '',
                 verbose: bool = False,
                 time_fn: Callable[[], float] = time.perf_counter,
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

        
    def reset(self) -> None:
        """
        Reset all attributes regardless of whether timer is running.
        """
        self._t_start = None
        self._start_included = None
        self._t_stop = None
        self._stop_included = None
        self._running = False
        self._timestamps = []
        self._stats = None
                        
                    
    def start(self, include: bool = False) -> float:
        """
        Start the timer. Set ``tic=True`` to include to start
        the timestamp array with a zero.
        
        Returns 0.
        
        """
        self.check_not_running()
        self._running = True
        self._t_start = self._time_fn()
        if include:
            self._start_included = True
            self._timestamps = [0]
        else:
            self._start_included = False
        return 0
            
            
    def tic(self) -> float:
        self.check_running()
        t = self._time_fn() - self._t_start
        self._timestamps.append(t)
        return self


    def stop(self, include: bool = False) -> float:
        """
        Stop the timer. Set ``tic=True`` to append timestamps
        with stop time. Prints a summary if verbose.
        """
        self.check_running()
        self._running = False
        self._t_stop = self._time_fn() - self._t_start
        if include:
            self._stop_included = True
            self._timestamps.append(self._t_stop)
        else:
            self._stop_included = False
            
        self._timestamps = np.array(self._timestamps)
        
        if self._verbose:
            self.print_summary()


    def check_running(self) -> None:
        """Raise ``RuntimeError`` if not running."""
        if not self._running:
            raise RuntimeError('Timer is not running.')


    def check_not_running(self) -> None:
        """Raise ``RuntimeError`` if running."""
        if self._running:
            raise RuntimeError('Timer is running.')


    def print_summary(self) -> None:
        """Print some timestamps info."""


        if self.name:
            s = f"Timer('{self.name}')\n"
        else:
            s = "     Timer     \n"
        s += '-' * (len(s) - 1) + '\n'

        ts = self.timestamps        
        n_tics = len(ts)
        t_stop = self._t_stop
        tics_per_sec = n_tics / t_stop
        periods = np.ediff1d(ts)
        
        s += f'time: {t_stop}\n'      
        s += f'tics: {n_tics}\n'
        s += f'tics/sec: {tics_per_sec}\n'
        for stat_name in ('mean', 'median', 'min', 'max'):
            fn = getattr(np, stat_name)
            stat = fn(periods)
            s += f'{stat_name}: {stat}\n'
                        
        print(s, flush=True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
