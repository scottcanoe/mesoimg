import sys
import time
import zmq
from zmq.utils.monitor import recv_monitor_message
from mesoimg.common import *
from mesoimg.messaging import *



HOST = '*'
PORT = 6000
topic = ''

def send(data: str, topic='a') -> None:
    """
    Send bytes with a socket.
    """
    sock.send_string(topic, zmq.SNDMORE)
    sock.send_string(data)
    time.sleep(0.01)


def close() -> None:
    sock.close()
    ctx.term()
    sys.exit(0)


EVENT_MAP = {}
print("Event names:")
for name in dir(zmq):
    if name.startswith('EVENT_'):
        value = getattr(zmq, name)
        print("%21s : %4i" % (name, value))
        EVENT_MAP[value] = name


def event_monitor(monitor):
    while monitor.poll():
        evt = recv_monitor_message(monitor)
        evt.update({'description': EVENT_MAP[evt['event']]})
        print("Event: {}".format(evt))
        if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
            break
    monitor.close()
    print("event monitor thread done!")


ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REP)
monitor = sock.get_monitor_socket()

sock.bind(f'tcp://{HOST}:{PORT}')

