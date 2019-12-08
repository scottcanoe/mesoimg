from pathlib import Path

def init():
    userdir = Path.home() / '.mesoimg'
    if not userdir.exists():
        userdir.mkdir()



init()