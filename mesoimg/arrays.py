from typing import Optional, Union
import numpy as np



__all__ = [
    'Frame',
]



class Frame(np.ndarray):

    """
    Array with attributes (i.e., a `__dict__`).

    """

    _index: Optional[int] = None
    _timestamp: Optional[float] = None

    def __new__(cls,
                data: np.ndarray,
                dtype: Optional[Union[str, type]] = None,
                copy: bool = True,
                **attrs,
                ):


        # Copy input 'data', and make it frame type.
        self = np.asarray(data).view(cls)

        # Make sure dimensions are reasonable.
        if self.ndim < 2:
            raise TypeError('too few dimensions for frame object.')

        # If initializing from another frame, assume its attributes.
        if isinstance(data, Frame):
            self.__dict__.update(data.__dict__)

        # Update with attributes sent as keyword args.
        for key, val in attrs.items():
            setattr(self, key, val)

        # Finally, return the instance.
        return self


    @property
    def index(self) -> Optional[int]:
        return self._index

    @index.setter
    def index(self, val: Optional[int]) -> None:
        assert isinstance(val, int) or val is None
        self._index = val

    @property
    def ix(self) -> Optional[int]:
        """Alias for index."""
        return self.index

    @ix.setter
    def ix(self, val: Optional[int]):
        self.index = val

    @property
    def timestamp(self) -> Optional[float]:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, val: Optional[float]):
        assert np.isreal(val) or val is None
        self._timestamp = val

    @property
    def ts(self) -> Optional[float]:
        """Alias for 'timestamps' """
        return self.timestamp

    @ts.setter
    def ts(self, val: Optional[float]):
        self.timestamp = val


    #--------------------------------------------------------------------------#
    # Public methods


    def contiguous(self) -> bool:
        return self.flags.c_contiguous


    def ascontiguous(self, in_place: bool = False) -> Optional['Frame']:

        if self.contiguous:
            if in_place:
                return
            return Frame(self, **self.__dict__)

        data = np.ascontiguousarray(self)
        if in_place:
            self[:] = data
            return
        return Frame(data, **self.__dict__)


    #--------------------------------------------------------------------------#
    # Class methods


    @classmethod
    def empty(self, *args, **kw) -> 'Frame':
        """Create a frame populated with all zeros."""
        return Frame(np.empty(*args, **kw))


    @classmethod
    def empty_like(self, *args, **kw) -> 'Frame':
        return Frame(np.empty_like(*args, **kw))


    @classmethod
    def ones(self, *args, **kw) -> 'Frame':
        """Create a frame populated with all ones."""
        return Frame(np.ones(*args, **kw))

    @classmethod
    def ones_like(self, template: np.ndarray, **kw) -> 'Frame':

        return Frame(np.ones_like(*args, **kw))


    @classmethod
    def zeros(self, *args, **kw) -> 'Frame':
        """Create a frame populated with all zeros."""
        return Frame(np.zeros(*args, **kw))


    @classmethod
    def zeros_like(self, *args, **kw) -> 'Frame':

        return Frame(np.zeros_like(*args, **kw))


    #--------------------------------------------------------------------------#
    # Private methods


    def __array_finalize__(self, obj):

        # Handle arriving from explicit constructor call.
        if obj is None:
            return

        # Handle arriving from  view casting or slicing. This is where the new
        # object would get attributes carried over from the original.
        if isinstance(obj, Frame):
            self.__dict__.update(obj.__dict__)
            return


    def __getitem__(self, key):

        data = np.ndarray.__getitem__(self, key)
        ndim = np.ndim(data)
        if ndim == 0:
            return data
        if ndim == 1:
            return data.view(np.ndarray)
        return data


    def __repr__(self) -> str:
        s  = f'<Frame: shape={self.shape}, '
        s += f'index={self._index}, '
        if hasattr(self, 'label'):
            if self.label is None:
                s += f'label=None, '
            else:
                s += f"label='{self.label}', "
        s += f'timestamp={self._timestamp}>'
        return s


