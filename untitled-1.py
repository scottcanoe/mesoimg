from collections import UserDict
import json


class Property:
    
    def __init__(self, key):
        self.key = key
    
    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        return obj[self.key]
    
    def __set__(self, obj, val):
        obj[self.key] = val
    
    def __del__(self, obj):
        del obj[self.key]



class CamConfig(UserDict):
    
    #: Width and height of imagery in pixels.
    resolution = Property('resolution')
    
    #: Limiting framerate.
    framerate = Property('framerate')
    
    #: Sensor mode indicates things like resolution, binning, etc.
    sensor_mode = Property('sensor_mode')
    
    #: Whether to use RGB data rather than compressed representations.
    raw = Property('raw')
    
    #: If in raw mode, define which channels to be discarded (if any).
    channels = Property('channels')
    
    #: Analog gain controls.
    exposure_mode = Property('exposure_mode')
    
    #: Last frame readout time (actually line readout time * num. lines). 
    shutter_speed = Property('shutter_speed')
    
    #: White-balance mode.
    awb_mode = Property('awb_mode')
    
    #: Two-tuple of white-balance gains (red, blue).
    awb_gains = Property('awb_gains')
    
    #: Whether to horizontally flip image (in GPU).
    hflip = Property('hflip')
    
    #: Whether to vertically flip image (in GPU).
    vflip = Property('vflip')
    
    #: Which attributes to keep records of when recording video.
    signals = Property('signals')
           
           
    def __init__(self, name=None, **kw):
        
        # Initialize dict with defaults.
        self.data = _CONFIGS['default'].copy()
        
        # Update from hard-coded settings or settings stored in files.
        if name is not None:

            if name in _CONFIGS.keys():
                self.data.update(_CONFIGS[name])
                        
            else:
                path = name
                with open(path, 'r') as f:
                    aux = json.load(f)
                self.data.update(aux)
        
        # Finally, let any supplied keyword args override other values.
        self.data.update(kw)

        
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f)


# Define default config.
#----------------------------------

_CONFIGS = {}
_CONFIGS['default'] = {
    'resolution'    : None,
    'framerate'     : None,
    'sensor_mode'   : 0,
    'raw'           : False,
    'channels'      : 'rgb',
    'exposure_mode' : 'auto',
    'shutter_speed' : 0,
    'awb_mode'      : 'auto',
    'awb_gains'     : (0, 0),
    'hflip'         : False,
    'vflip'         : False,
    'signals'       : [],
}


# Define raw image/video formats.
#----------------------------------

_CONFIGS['rgb'] = {
    'resolution'    : (640, 480),
    'framerate'     : 30.0,
    'sensor_mode'   : 7,
    'raw'           : True,
    'channels'      : 'rgb',
    'exposure_mode' : 'fix',
    'awb_mode'      : 'fix',
}

for channel in ('r', 'g', 'b'):
    d = _CONFIGS['rgb'].copy()
    d['channels'] = channel
    _CONFIGS[channel] = d




