import ast
import contextlib
import io
import sys
import traceback
from typing import (Any,
                    Callable,
                    Dict,
                    Iterable,
                    Mapping,
                    NamedTuple,
                    Optional,
                    Tuple)


__all__ = [
    'infer_target',
    'infer_efun',
    'execute',
]


DEFAULT_TARGETS = (\
    'cam.',
    'self.',
    'server.',
)


def infer_target(expr: str,
                 targets: Iterable[str] = DEFAULT_TARGETS,
                 default: str = '',
                 ) -> str:

    # Find the prefix to determine which computer will do the evaluation.
    for t in targets:
        if expr.startswith(t):
            return t
    return default


def infer_efun(expr: str) -> Callable:

    # Determine whether to use `eval` or `exec`.
    for node in ast.walk(ast.parse(expr)):
        if isinstance(node, ast.Expr):
            return eval
    return exec


def execute(expr: str,
            efun: Optional[Callable] = None,
            globals_: Optional[Dict] = None,
            locals_ : Optional[Mapping] = None,
            suppress: bool = True,
            ) -> Tuple[Any, str, str]:
    """

    Returns
    -------

    result:
       The result of `eval`/`exec` (always `None` when using `exec`).

    out: str
        The redirected output, often the empty string.

    err: str
        If `supress` is `True`, exceptions will be caught and formatted
        as a display-ready string. If no exceptions are raised, the
        empty string is returned.

    """
    efun = infer_efun(expr) if efun is None else efun
    globals_ = globals() if globals_ is None else globals_
    locals_ = locals() if locals_ is None else locals_
    result, out, err = None, io.StringIO(), ''
    with contextlib.redirect_stdout(out):
        if suppress:
            try:
                result = efun(expr, globals_, locals_)
            except:
                err = ''.join(traceback.format_exception(*sys.exc_info()))

    return result, out.getvalue(), err

