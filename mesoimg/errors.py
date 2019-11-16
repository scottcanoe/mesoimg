# Errors.


class StopRecording(RuntimeError):
    pass


class FileIsFull(StopRecording):
    pass
