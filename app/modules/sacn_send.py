import sacn
import atexit
import array
import numpy as np
from typing import Callable, Optional, List, Dict, Union

class SacnSend:
    def __init__(self, bind_address = "127.0.0.1", universe_count=1, multicast=True, dummy=False):
        self.multi = multicast
        self.sender = sacn.sACNsender(bind_address)
        self.sender.activate_output(universe)
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
            chunk_data = array.array('B')
            for j in range(0, len(chunk), 3):
                chunk_data.append(chunk[j])
            dmx_data.append(chunk_data)
        return dmx_data

    def send_sacn_data(self, data: List[List[int]]):
        for i in range(len(data)):
            self.sender[i+1].dmx_data = data[i]


    def send_frame(self, frame: np.array):
        data = self.convert_frame_to_sacn_data(frame)
        self.send_sacn_data(data)
