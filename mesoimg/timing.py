from collections import OrderedDict
import logging
from typing import Any, Callable, Optional
import time
from time import perf_counter
import numpy as np


__all__ = [
    'Clock',
    'master_clock',
    'CountdownTimer',
    'IntervalTimer',
    'repr_secs',
]



class Clock:

    """
    Clock/timer with accurate (sub-millisecond) accuracy.
    Clock's "t-zero" is set when instantiated.
    """

    def __init__(self, can_reset: bool = True):
        self._can_reset = can_reset
        self._t_start = perf_counter()

    @property
    def can_reset(self) -> bool:
        return self._can_reset

    def reset(self) -> None:
        if self._can_reset:
            self._t_start = perf_counter()
        else:
            raise TypeError("Clock instance cannot be reset.")

    def __call__(self):
        """
        Get the clock's current time.
        """
        return perf_counter() - self._t_start


master_clock = Clock(can_reset=False)



class CountdownTimer:

    def __init__(self, duration: float):
        self._duration = duration
        self._t_start = perf_counter()

    @property
    def duration(self) -> float:
        return self._duration

    def reset(self) -> None:
        self._t_start = perf_counter()

    def __call__(self):
        """
        Get the clock's remaining time.
        """
        return self._duration - (perf_counter() - self._t_start)



class IntervalTimer:

    """
    Class for ticking off intervals.
    """

    def __init__(self,
                 name: str = '',
                 start: bool = True,
                 verbose: bool = False,
                 time_fn: Callable[[], float] = time.perf_counter,
                 logger: Optional[logging.Logger] = None):

        self._name = name
        self._verbose = verbose
        self._time_fn = time_fn
        self._logger = logger
        self.reset()
        if start:
            self.start()

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

    @property
    def count(self) -> int:
        return len(self._timestamps)


    def reset(self) -> None:
        """
        Reset all attributes regardless of whether timer is running.
        """
        self._t_start = None
        self._t_stop = None
        self._running = False
        self._timestamps = []


    def start(self) -> float:
        """
        Start the timer. Does not add a timestamp.
        """
        t = self._time_fn()
        self._check_not_running()
        self._running = True
        self._t_start = t
        return 0


    def tic(self) -> float:
        """
        Add a timestamp.
        """
        t = self._time_fn() - self._t_start
        self._check_running()
        self._timestamps.append(t)
        return self


    def stop(self) -> float:
        """
        Stop the timer. Set ``tic=True`` to append timestamps
        with stop time. Prints a summary if verbose.
        """
        t = self._time_fn() - self._t_start
        self._check_running()
        self._running = False
        self._t_stop = t
        self._timestamps = np.array(self._timestamps)
        if self._verbose:
            self.print_summary()


    def tell(self) -> float:
        """
        Get the time without adding a timestamp.
        """
        return self._time_fn() - self._t_start


    def _check_running(self) -> None:
        """Raise ``RuntimeError`` if not running."""
        if not self._running:
            raise RuntimeError('Timer is not running.')


    def _check_not_running(self) -> None:
        """Raise ``RuntimeError`` if running."""
        if self._running:
            raise RuntimeError('Timer is running.')


    def print_summary(self) -> None:
        """Print some timestamps info."""


        ts = self.timestamps
        n_tics = len(ts)
        n_secs = self._t_stop
        tics_per_sec = n_tics / n_secs
        periods = np.ediff1d(ts)

        if self.name:
            s = f"Timer('{self.name}')\n"
        else:
            s = "     Timer     \n"
        s += '-' * (len(s) - 1) + '\n'


        s += 'time: {:.2f} {}\n'.format(*repr_secs(n_secs))
        s += f'tics: {n_tics}\n'
        s += 'tics/sec: {:.2f}\n'.format(repr_secs(tics_per_sec)[0])
        for stat_name in ('mean', 'median', 'min', 'max'):
            stat = getattr(np, stat_name)(periods)
            s += '{} ITI: {:.2f} {}\n'.format(stat_name, *repr_secs(stat))

        print(s, flush=True)




def repr_secs(secs: float):
    """
    Format duration into a string, including appropriate units.
    """

    sign, secs = np.sign(secs), np.abs(secs)
    if secs >= 1:
        return sign * secs, 'sec.'
    if secs >= 1e-3:
        return sign * secs * 1e3, 'msec.'
    elif secs >= 1e-6:
        return sign * secs * 1e6, 'usec.'
    else:
        return sign * secs * 1e9, 'nsec'


