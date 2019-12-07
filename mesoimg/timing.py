from typing import Any, Callable, Tuple, Optional
import time
import numpy as np


__all__ = [
    'Clock',
    'CountdownClock',
]



class Clock:

    """
    Clock/timer with accurate (sub-millisecond) accuracy.
    Clock's "t-zero" is set when instantiated.
    """

    def __init__(self,
                 can_reset: bool = True,
                 time_fn: Callable[[], float] = time.perf_counter,
                 ):

        self._can_reset = can_reset
        self._time_fn = time_fn
        self._datum = self._time_fn()


    @property
    def can_reset(self):
        return self._can_reset

    def reset(self) -> None:
        if not self.can_reset:
            raise AttributeError("clock cannot be reset.")
        self._datum = self._time_fn()


    def __call__(self) -> float:
        """
        Get the clock's current time.
        """
        return self._time_fn() - self._datum



class CountdownClock(Clock):

    def __init__(self, duration: float, **kw):
        super().__init__(**kw)
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self) -> float:
        """
        Get the clock's remaining time. May be negative.
        """
        return self._duration - (self._time_fn() - self._datum)







class EventLogger:




    """
    Log event times.
    """



    def __init__(self,
                 ID: Optional[Any] = None,
                 can_reset: bool = True,
                 time_fn: Callable[[], float] = time.perf_counter,
                 start: bool = False,
                 ):

        self.ID = ID
        self._can_reset = can_reset
        self._time_fn = time_fn
        self._datum = self._time_fn()
        self._entries = []


    @property
    def can_reset(self):
        return self._can_reset

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(self._timestamps)


    @property
    def entries(self):
        return tuple(self._entries)

    @property
    def count(self) -> int:
        return len(self._timestamps)

    @property
    def timestamps(self) -> np.ndarray:
        return np.ndarray([e.timestamp for e in self._entries])


    def reset(self) -> None:
        """
        Reset all attributes regardless of whether timer is running.
        """
        if not self._can_reset:
            raise AttributeError("event logger cannot be reset.")
        self._datum = self._time_fn()
        self._entries = []


    def log(self, event: Optional[Any] = None) -> float:
        """
        Log an event.
        """

        timestamp = self._time_fn() - self._datum
        self._entries.append(Entry(timestamp=timestamp, event=event))
        return timestamp



    def tell(self) -> float:
        """
        Get the time without adding a timestamp.
        """
        return self._time_fn() - self._t_start



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



