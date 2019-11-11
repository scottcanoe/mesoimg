import numpy as np

def test_dtype():
    
    """
    Show that endianness doesn't matter for uint8
    since bit order is apparently the the same for both.
    Is this because bit order is platform determined?
    """
    u1 = np.dtype('>u1')
    u2 = np.dtype('<u1')
    
    dat = [1, 3, 5, 2, 7, 0, 2]
    a1 = np.array(dat, u1)
    a2 = np.array(dat, u2)
    
    b1 = a1.tobytes()
    b2 = a2.tobytes()
    
    m1 = memoryview(a1)
    m2 = memoryview(a2)
    
    c1 = m1.tobytes()
    c2 = m2.tobytes()
    
    assert u1 == u2
    


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])