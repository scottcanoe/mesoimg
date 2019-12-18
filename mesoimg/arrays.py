from numbers import Number
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.lib.mixins


__all__ = [
    'Frame',
]



class Frame(numpy.lib.mixins.NDArrayOperatorsMixin):

    _data: np.ndarray
    _index: Optional[int] = None
    _timestamp: Optional[float] = None

    def __init__(self,
                 obj: Any,
                 copy: bool = True,
                 dtype: Optional[type] = None,
                 **attrs,
                 ):

        if isinstance(obj, Frame):
            Frame._init_from_frame(self, obj, copy, dtype, **attrs)

        elif isinstance(obj, np.ndarray):
            Frame._init_from_array(self, obj, copy, dtype, **attrs)

        else:
            self._data = np.array(obj, dtype=dtype)
            self.index = attrs.get('index')
            self.timestamp = attrs.get('timestamp')

    @property
    def data(self) -> np.array:
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self) -> Tuple[int]:
        return self._data.shape

    @property
    def flags(self) -> np.ndarray:
        return self._data.flags

    @property
    def index(self) -> Optional[int]:
        return self._index

    @property
    def ix(self) -> Optional[int]:
        """Convenience accessor for `index`."""
        return self._index

    @index.setter
    def index(self, val: Optional[int]) -> None:
        assert isinstance(val, int) or val is None
        self._index = val

    @property
    def timestamp(self) -> Optional[float]:
        return self._timestamp

    @property
    def ts(self) -> Optional[float]:
        """Convenience accessor for `timestamp`"""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, val: Optional[float]) -> None:
        assert np.isreal(val) or val is None
        self._timestamp = val

    @property
    def contiguous(self) -> bool:
        return self.flags.c_contiguous

    #--------------------------------------------------------------------------#
    # Public methods


    def astype(self, dtype: type) -> 'Frame':
        return Frame(self, dtype=dtype)


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
    def ones(self,
             shape: Sequence[int],
             dtype: Optional[type] = None,
             **attrs,
             ) -> 'Frame':
        """Create a frame populated with all ones."""
        return Frame(np.ones(shape, dtype=dtype), **attrs)

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


    @classmethod
    def _init_from_frame(cls,
                         self: 'Frame',
                         obj: 'Frame',
                         copy: bool,
                         dtype: Optional[type],
                         **attrs,
                         ) -> None:

        data = obj._data
        if copy or dtype and dtype != data.dtype:
            data = np.array(data, dtype=dtype)
        self._data = data
        self.__dict__.update(obj.__dict__.copy())

        # Set attributes from keyword args.
        if 'index' in attrs:
            self.index = attrs.pop('index')
        if 'timestamp' in attrs:
            self.timestamp = attrs.pop('timestamp')
        self.__dict__.update(attrs)



    @classmethod
    def _init_from_array(cls,
                             self: 'Frame',
                             data: np.ndarray,
                             copy: bool,
                             dtype: type,
                             **attrs,
                             ) -> None:

        if copy or dtype and dtype != data.dtype:
            data = np.array(data, dtype=dtype)
        self._data = data

        # Set attributes from keyword args.
        if 'index' in attrs:
            self.index = attrs.pop('index')
        if 'timestamp' in attrs:
            self.timestamp = attrs.pop('timestamp')
        self.__dict__.update(attrs)


    def __getitem__(self, key):
        data = self._data[key]
        return data


    def __setitem__(self, key, val: Any) -> None:
        self._data[key] = val


    def __array__(self):
        return self._data


    def __array_ufunc__(self, ufunc, method, *args, **kw):

        if method == '__call__':
            new_args = []
            for obj in args:

                if isinstance(obj, Frame):
                    obj = obj._data
                elif isinstance(obj, Number):
                    obj = np.ones_like(self._data) * obj
                new_args.append(obj)
            return self.__class__(ufunc(*new_args, **kw))
        else:
            return NotImplemented

    def __iadd__(self, other: Union[Number, np.ndarray, 'Frame']) -> 'Frame':
        self._data += other
        return self

    def __isub__(self, other: Union[Number, np.ndarray, 'Frame']) -> 'Frame':
        self._data -= other
        return self

    def __imul__(self, other: Union[Number, np.ndarray, 'Frame']) -> 'Frame':
        self._data *= other
        return self

    def idiv(self, other) -> 'Frame':
        """
        Couldn't get ufuncs or operator overloads to work with this one.
        """
        if isinstance(other, Frame):
            other = other._data
        self._data /= other
        return self

    def ifloordiv(self, other) -> 'Frame':
        """
        Couldn't get ufuncs or operator overloads to work with this one.
        """
        if isinstance(other, Frame):
            other = other._data
        self._data //= other
        return self

    def __repr__(self) -> str:
        s  = f'<Frame: shape={self.shape}, '
        s += f'index={self._index}, '
        if hasattr(self, 'label'):
            if self.label is None:
                s += f'label=None, '
            else:
                s += f"label='{self.label}', "
        s += f'timestamp={self._timestamp:.4f}>'
        return s



if __name__ == '__main__':

    #f = Frame.ones([2, 2])
    a = Frame.ones([2, 2], index=0, timestamp=0)
    #b = Frame.ones([2, 2]) + 1
    #np.mean(a)
    #c = np.multiply(a, a)
    #a -= 1
    #a /= 1
