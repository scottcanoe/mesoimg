import abc
from typing import Any, ClassVar, Dict, List, Optional, Tuple


__all__ = [
    'RequestError',
    'Request',
    'Get',
    'Set',
    'Call',
    'Exec',
    'Response',
]


class RequestError(RuntimeError):
    pass


class Request(abc.ABC):

    _action: ClassVar[str] = ''
    _fields: ClassVar[Tuple] = tuple([])


    def asdict(self) -> Dict[str, Any]:
        d = {name : getattr(self, name) for name in self._fields}
        d['action'] = self._action
        return d

    def __repr__(self) -> str:
        parts = [f'{name}={getattr(self, name)}' for name in self._fields]
        return f'<{self._action.upper()}: ' + ', '.join(parts) + '>'


    @staticmethod
    def from_json(d: Dict) -> 'Request':

        # Validate 'action'.
        try:
            action = d.pop('action')
        except KeyError:
            raise RequestError('request has no action.')

        try:
            cls = _REQUEST_CLASSES[action]
            req = cls(**d)
        except KeyError:
            d['action'] = action
            raise RequestError(f'invalid action: {action}')

        d['action'] = action
        return req



class Get(Request):

    _action: ClassVar[str] = 'get'
    _fields = ('key',)

    def __init__(self, key: str):
        self.key = key



class Set(Request):

    _action: ClassVar[str] = 'set'
    _fields = ('key', 'val')

    def __init__(self, key: str, val: Any):
        self.key = key
        self.val = val


class Call(Request):

    _action: ClassVar[str] = 'call'
    _fields = ('key', 'args', 'kw')

    def __init__(self,
                 key: str,
                 args: Optional[List] = None,
                 kw: Optional[Dict] = None,
                 ):

        self.key = key
        self.args = args if args else []
        self.kw = kw if kw else {}


class Exec(Request):

    _action: ClassVar[str] = 'exec'
    _fields = ('text',)

    def __init__(self, text: str):
        self.text = text



_REQUEST_CLASSES = {\
    'get'  : Get,
    'set'  : Set,
    'call' : Call,
    'exec' : Exec,
}



class Response:

    def __init__(self,
                 result: Any = None,
                 stdout: str = '',
                 error: str = '',
                 ):

        self.result = result
        self.stdout = stdout
        self.error = error


    def __repr__(self) -> str:
        return f'<Response: result={self.result}, ' + \
               f'stdout={self.stdout}, ' + \
               f'error={self.error}>'
