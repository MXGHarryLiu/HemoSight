import json
import os
import socket

current_dir = os.path.dirname(__file__)

class GlobalSettings:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(GlobalSettings, cls).__new__(cls)
            cls._debug = None
        return cls._instance
    
    def __init__(self):
        self.debug = False

    @property
    def debug(self) -> bool:
        return self._debug
    
    @debug.setter
    def debug(self, value: bool) -> None:
        if self._debug == value:
            return
        path = os.path.join(current_dir, 'settings.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Settings file not found: {path}')
        if value:
            # obtain system name
            name = socket.gethostname()
            path = os.path.join(current_dir, f'settings_{name}.json')
            if not os.path.exists(path):
                print(f'Debug mode cannot be turned on because no settings_{name}.json.')
                value = False
                path = os.path.join(current_dir, 'settings.json')
        if value:
            print(f'Debug mode is on. Using settings_{name}.json.')
        elif self._debug is not None:
            print('Debug mode is off. Using settings.json.')
        self._debug = value
        self._setting_file = path
        return

    def get(self, key: str):
        with open(self._setting_file, 'r') as file:
            # load key from json
            data = json.load(file)
            return data[key]
    