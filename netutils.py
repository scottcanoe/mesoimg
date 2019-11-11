"""
Various networking utilities.
"""
import fnmatch
from typing import List, Union, Optional
import psutil


def kill(p: Union[int, psutil.Process]) -> None:
    """
    Kill a process.
    """

    p = psutil.Process(p) if isinstance(p, int) else p
    p.kill()


def net_connections(kind: str = 'inet',
                    *,
                    ip: Optional[str] = None,
                    port: Optional[int] = None,                    
                    ) -> List[psutil._common.sconn]:

    """
    Like psutil.net_connections but with built-in filtering capabilities.
    If `ip` and/or `port` are supplied, only connections with a
    matching endpoint will be returned. Wildcards `ip` addresses
    are supported.
    
    """

    def ip_match(addr):
        return addr and fnmatch.fnmatch(addr.ip, ip)

    def port_match(addr):
        return addr and addrport == port


    # Get all network connections.
    conns = psutil.net_connections(kind)
        
    # Apply ip filter.
    if ip:
        conns = [c for c in conns if ip_match(c.laddr) or \
                                     ip_match(c.raddr)]
            
    # Apply port filter.
    if port:
        conns = [c for c in conns if port_match(c.laddr) or \
                                     port_match(c.raddr)]

    return conns


def net_pids(*args, **kw) -> List[int]:

    """
    Return pids that are associated with network connections.    
    See `net_connections` for meaning of ``*args`` and ``**kw``.    
    """
    
    conns = net_connections(*args, **kw)
    return list(set([c.pid for c in conns if c.pid is not None]))



def net_processes(*args, **kw) -> List[psutil.Process]:
    """
    Return processes objects associated with network connections.
    See `net_connections` for meaning of ``*args`` and ``**kw``.
    
    """
    return [psutil.Process(pid) for pid in net_pids(*args, **kw)]

