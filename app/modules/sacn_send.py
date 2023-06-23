import sacn
import atexit
import array
import logging
import numpy as np
from typing import Callable, Optional, List, Dict, Union

class SacnSend:
    def __init__(self, bind_address = "127.0.0.1", universe_count=1, multicast=True, dummy=False, brightness = 50.0):
        self.multi = multicast
        self.sender = sacn.sACNsender(bind_address)
        self.sender.fps = 60
        self.brightness = brightness

        for i in range(1, universe_count + 1):
            if not dummy:
                self.sender.activate_output(i)  # start sending out data in the 1st universe
            if self.multi:
                self.sender[i].multicast = True

        self.sender.start()
        atexit.register(self.sender.stop)


    def convert_frame_to_sacn_data(self, frame: np.array) -> List[List[int]]:
        # Convert WW animation frame to sACN data format
        dmx_data = []
        for i in range(0, len(frame), 510):
            chunk = frame[i:i+510]
            dmx_data.append(list(chunk))
        return dmx_data

    def send_sacn_data(self, data: List[List[int]]):
        for i in range(len(data)):
            #  # scale data by brightness
            #  scaled_data = [round(byte * float(self.brightness / 255.0)) for byte in data[i]]
            #  self.sender[i+1].dmx_data = scaled_data
            self.sender[i+1].dmx_data = data[i]

    def send_frame(self, frame: np.array):
        data = self.convert_frame_to_sacn_data(frame)
        self.send_sacn_data(data)
