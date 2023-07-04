import sacn
import atexit
import array
import logging
import numpy as np
import math 

from typing import Callable, Optional, List, Dict, Union

class SacnSend:
    def __init__(self, bind_address = "127.0.0.1", num_strips=1, num_pixels=1, multicast=True, dummy=False, brightness = 50.0):
        self.multi = multicast
        self.sender = sacn.sACNsender(bind_address)
        self.sender.fps = 60
        self.brightness = brightness
        self.num_strips = num_strips
        self.num_pixels = num_pixels
        self.universe_count = math.ceil( (num_strips * num_pixels * 3) / 510)
        logging.info(f"sACN Number of pixels: {self.num_pixels}, Number of strips: {self.num_strips}, Universe count: {self.universe_count}")

        for i in range(1, self.universe_count + 1):
            if not dummy:
                self.sender.activate_output(i) 
            if self.multi:
                self.sender[i].multicast = True

        self.sender.start()
        atexit.register(self.sender.stop)

    def convert_frame_to_sacn_data(self, frame: np.array) -> List[List[int]]:
        np_frame = np.frombuffer(frame, dtype=np.uint8)
        scaled_brightness = self.brightness / 255
        scaled_frame = (np_frame * scaled_brightness).astype(np.uint8)
        scaled_frame = scaled_frame.tobytes()

        dmx_data = []  # List to store the DMX data
        universe_count = 1  # Variable to keep track of the universe count
        channel_count = 1  # Variable to keep track of the channel count

        for strip in range(self.num_strips):
            strip_universes = math.ceil(self.num_pixels / 170)  # Calculate the number of universes needed for the current strip

            for _ in range(strip_universes):
                remaining_pixels = self.num_pixels - ((channel_count - 1) // 3)  # Calculate the remaining pixels for the current universe
                universe = universe_count  # Store the current universe

                if remaining_pixels >= 170:
                    # If there are enough pixels to fill the universe, append a list of 512 values representing the universe count
                    dmx_data.append([universe_count] * 510)
                    channel_count += 510  # Increment the channel count by the number of channels used in the universe
                else:
                    # If there are not enough pixels to fill the universe, create a list with the remaining pixels
                    universe_data = [universe_count] * (remaining_pixels * 3)

                    # Append the universe data to the DMX data list
                    dmx_data.append(universe_data)
                    channel_count += (remaining_pixels * 3)  # Increment the channel count by the number of channels used by the remaining pixels

                logging.info(f"Strip: {strip + 1}, Universe: {universe}, Channel: {channel_count - (remaining_pixels * 3) - 1}")
                universe_count += 1  # Increment the universe count for the next strip

        return dmx_data

    def send_sacn_data(self, data: List[List[int]]):
        for i in range(len(data)):
            self.sender[i+1].dmx_data = data[i]
            #  # scale data by brightness
            #  scaled_data = [round(byte * float(self.brightness / 255.0)) for byte in data[i]]
            #  self.sender[i+1].dmx_data = scaled_data

    def send_frame(self, frame: np.array):
        data = self.convert_frame_to_sacn_data(frame)
        self.send_sacn_data(data)
