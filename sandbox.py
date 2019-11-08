import io
import logging
import time
import numpy as np
import picamera
from mesoimg import MesoCam
from mesoimg.common import *


logging.basicConfig(level=logging.INFO)



def record(cam=None, *, max_frames=None, max_secs=None):

    # Handle max_frames and max_secs arguments.
    max_frames = np.inf if max_frames is None else max_frames
    max_secs = np.inf if max_secs is None else max_secs
    if max_frames == np.inf and max_secs == np.inf:
        raise ValueError('max_frames and max_secs cannot both be infinite.')

    # Setup state and record variables.
    frame_counter = 0
    frames = []
    clock = Clock()
    timestamps = []
    stream = io.BytesIO()

    if cam is None:
        cam = MesoCam()

    logging.info('Beginning continuous capture.')
    try:
        clock.start()
        for foo in cam.capture_continuous(stream, 'rgb', use_video_port=True):

            frame_counter += 1
            ts = clock.time()

            if frame_counter > max_frames or ts > max_secs:
                break

            # Grab the buffered data, and remove non-green channels.
            arr = stream.getvalue()
            stream.seek(0)

            # Keep only green chanel.
            arr = arr[1::3]

            # Update frames and timestamps.
            frames.append(arr)
            timestamps.append(ts)

    except:
        cam.close()
        clock.stop()
        raise

    cam.close()
    clock.stop()

    MesoCam.print_timing_summary(timestamps)

    timestamps = np.array(timestamps)
    frames = [np.frombuffer(fm, dtype=uint8) for fm in frames]

    return timestamps, frames


#cam = MesoCam()
#cam.close()

T, frames = record(max_secs=5)











