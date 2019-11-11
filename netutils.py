"""
Various networking utilities.
"""
import fnmatch
from typing import Callable, List, Union, Optional
import psutil


__all__ = [
    'kill',
    'net_connections',
    'net_processes',
]


def kill(p: Union[int, psutil.Process]) -> None:
    """
    Kill a process.
    """

    p = psutil.Process(p) if isinstance(p, int) else p
    p.kill()



def net_connections(kind: str = 'inet',
                    ip: Optional[str] = None,
                    port: Optional[int] = None,
                    status: Optional[str] = None) \
                    -> List[psutil._common.sconn]:

    """
    Like psutil.net_connections but with built-in filtering capabilities.
    Optionally filter by `kind` (e.g., 'tcp'), `ip`, `port`, or `status`.
    `ip` may contain wildcards.
    
    See https://psutil.readthedocs.io/en/latest/#connections-constants for
    status values. Not case-sensitive.

    """

    # Get all network connections.
    conns = psutil.net_connections(kind)
        
    # Apply ip filter.
    if ip:
        
        if '*' in ip:
            fn = lambda addr : fnmatch.fnmatch(addr, ip) if addr else False
        else:
            fn = lambda addr : addr == ip if addr else False        

        conns = [c for c in conns if fn(c.laddr) or fn(c.raddr)]
                                 
    # Apply port filter.
    if port:
        fn = lambda addr : addr.port == port if addr else False 
        conns = [c for c in conns if fn(c.laddr) or fn(c.raddr)]
 
    # Apply status filter.
    if status:
        status = status.upper()
        conns = [c for c in conns if c.status == status]
    
    return conns




def net_processes(name: Optional[str] = None,
                  kind: str = 'inet',
                  **kw) \
                  -> List[psutil.Process]:
    """
    Return processes objects associated with network connections.
    Optionally filter by process name (wildcards supported) and/or
    connection specs. See ``net_connections`` for those parameters.
    
    """

    # Handle filtering by connection specs if requested.
    if kw:
        conns = net_connections(kind=kind, **kw)
        pids = set([c.pid for c in conns if c is not None])
        procs = [psutil.Process(val) for val in pids]
    else:
        procs = [p for p in psutil.process_iter() if p.connections(kind=kind)]

    # Optionally filter by name.
    if name:
        if '*' in name:
            fn = lambda s : fnmatch.fnmatch(s, name)
        else:
            fn = lambda s : s == name
        procs = [p for p in procs if fn(p.name())]
        
    return procs
    

