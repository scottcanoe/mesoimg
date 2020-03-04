"""A thorough test of polling PAIR sockets."""

#-----------------------------------------------------------------------------
#  Copyright (c) 2010 Brian Granger
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file COPYING.BSD, distributed as part of this software.
#-----------------------------------------------------------------------------

import time
import zmq

print("Running polling tests for PAIR sockets...")

addr = 'tcp://127.0.0.1:5558'
ctx = zmq.Context()
s1 = ctx.socket(zmq.PAIR)
#s2 = ctx.socket(zmq.PAIR)

s1.bind(addr)
#s2.connect(addr)

# Sleep to allow sockets to connect.
time.sleep(1.0)

poller = zmq.Poller()
poller.register(s1, zmq.POLLIN|zmq.POLLOUT)
#poller.register(s2, zmq.POLLIN|zmq.POLLOUT)

## Now make sure that both are send ready.
#socks = dict(poller.poll(1))
#assert socks[s1] == zmq.POLLOUT
#assert socks[s2] == zmq.POLLOUT

## Now do a send on both, wait and test for zmq.POLLOUT|zmq.POLLIN
#s1.send(b'msg1')
#s2.send(b'msg2')
#time.sleep(1.0)
#socks = dict(poller.poll(1))
#assert socks[s1] == zmq.POLLOUT|zmq.POLLIN
#assert socks[s2] == zmq.POLLOUT|zmq.POLLIN

## Make sure that both are in POLLOUT after recv.
#m1 = s1.recv()
#m2 = s2.recv()
#socks = dict(poller.poll(1))
#assert socks[s1] == zmq.POLLOUT
#assert socks[s2] == zmq.POLLOUT

#poller.unregister(s1)
#poller.unregister(s2)

## Wait for everything to finish.
#time.sleep(1.0)

#print("Finished.")
