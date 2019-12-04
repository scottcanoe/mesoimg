from picamera import PiCamera


def opencam():
	return PiCamera(sensor_mode=7,
					resolution=(640, 480))


cam = opencam()


