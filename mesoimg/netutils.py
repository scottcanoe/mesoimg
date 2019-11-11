"""
Various networking utilities.
"""
import fnmatch as fnmatch
import os
import signal
from typing import Callable, List, Union, Optional
import psutil


__all__ = [
    'kill',
    'net_connections',
    'net_processes',
]


_WILD_CHARS = frozenset("*?[]!{}")

def fnmatcher(pat: str) -> Callable[[str], bool]:
    
    """
    Make a bool-valued string comparator for a given pattern.
    """
    
    if not _WILD_CHARS.isdisjoint(pat):
        return lambda s : fnmatch.fnmatch(s, pat)
    return lambda s : s == pat
    





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




def net_procs(name: Optional[str] = None,
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
        fn = fnmatcher(name)
        procs = [p for p in procs if fn(p.name())]
        
    return procs
    

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
        fn = fnmatcher(name)
        procs = [p for p in procs if fn(p.name())]
        
    return procs




def find_procs_by_name(name: str) -> List[psutil.Process]:
    "Return a list of processes matching a name or wildcard pattern."
    fn = fnmatcher(name)
    procs = []
    for p in psutil.process_iter(attrs=['name']):
        if fn(p.info['name']):
            procs.append(p)
    return procs
    


def kill(p: Union[int, psutil.Process]) -> None:
    """
    Kill a process.
    """

    p = psutil.Process(p) if isinstance(p, int) else p
    p.kill()



def kill_proc_tree(pid,
                   sig=signal.SIGTERM,
                   include_parent=True,
                   timeout=None,
                   on_terminate=None):
    
    """
    Kill a process tree (including grandchildren) with signal
    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
    called as soon as a child terminates.
    """
    
    assert pid != os.getpid(), "won't kill myself"
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    if include_parent:
        children.append(parent)
    for p in children:
        p.send_signal(sig)
    gone, alive = psutil.wait_procs(children,
                                    timeout=timeout,
                                    callback=on_terminate)
    return (gone, alive)    
    
    
def reap_children(timeout: float = 3):
    """
    Tries hard to terminate and ultimately kill all the children of this process.
    
    This may be useful in unit tests whenever sub-processes are started.
    This will help ensure that no extra children (zombies) stick around to hog
    resources.
    """

    def on_terminate(proc):
        print("process {} terminated with exit code {}".format(proc, proc.returncode))

    procs = psutil.Process().children()
    # send SIGTERM
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=timeout, callback=on_terminate)
    if alive:
        # send SIGKILL
        for p in alive:
            print("process {} survived SIGTERM; trying SIGKILL".format(p))
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(alive, timeout=timeout, callback=on_terminate)
        if alive:
            # give up
            for p in alive:
                print("process {} survived SIGKILL; giving up".format(p))