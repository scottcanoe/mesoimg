import zmq
from mesoimg.server import MesoServer





if __name__ == '__main__':
    ctx = zmq.Context()
    s = MesoServer(ctx)

    run_server()
