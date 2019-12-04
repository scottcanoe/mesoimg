from threading import Condition, Thread
import time
from time import sleep
from queue import Empty, Queue
import zmq



class ThreadedSocket(Thread):
	
	
	def __init__(self,
				 context: zmq.Context,
				 address: str,
				 timeout: float=1.0,
				 start: bool = False,
				 ):
		super().__init__()

		self.sock = ctx.socket(zmq.REP)
		self.sock.bind(address)
		self.poller = zmq.Poller()
		self.poller.register(self.sock, zmq.POLLIN|zmq.POLLOUT)		
		self.timeout = 1
		self._terminate = False
		if start:
			self.start()

	def run(self):

		while not self._terminate:
			socks = dict(self.poller.poll(self.timeout))
			if self.sock in socks:
				msg = self.sock.recv_string()
				print(f'Received message: {msg}')
				self.sock.send_string(f'From server: {msg}')
				sleep(0.1)

	def stop(self):
		self._terminate = True
		time.sleep(self.timeout + 0.1)
		self.poller.unregister(self.sock)
		self.sock.close()

	def close(self):
		"""Alias for stop()"""
		self.stop()


class Pusher(Thread):
	
	
	def __init__(self,
				 context: zmq.Context,
				 address: str,
				 q: Queue,				 
				 timeout: float = 1.0,
				 start: bool = False,
				 ):
		super().__init__()
		
		self.sock = ctx.socket(zmq.PUSH)
		self.sock.bind(address)
		self.poller = zmq.Poller()
		self.poller.register(self.sock, zmq.POLLOUT)		
		self.q = q
		self.timeout = 1		
		self._terminate = False
		if start:
			self.start()

	def run(self) -> None:

		while not self._terminate:
			try:
				msg = self.q.get(timeout=self.timeout)
			except Empty:
				continue			
			self.sock.send_string(msg)		
			time.sleep(0.05)

	
	def stop(self) -> None:
		self._terminate = True
		time.sleep(self.timeout + 0.1)
		self.poller.unregister(self.sock)
		self.sock.close()

	
	def close(self) -> None:
		"""Alias for Pusher.stop()"""
		self.stop()


q = Queue()

ctx = zmq.Context()
address = 'tcp://*:7000'
s = Pusher(ctx, address, q, start=True)
try:
	while True:
		msg = input('>> ')
		print(f'sending: {msg}')	
		q.put(msg)
except KeyboardInterrupt:
	s.close()
	ctx.term()



