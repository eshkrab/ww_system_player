import sacn
import atexit
import array
import logging
import numpy as np
import math 

from typing import Callable, Optional, List, Dict, Union, Tuple

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
        channel_count = 0  # Variable to keep track of the channel count

        for strip in range(self.num_strips):
            strip_pixels_counter = self.num_pixels # Number of pixels for the current strip
            # As long as we have pixels to arrange within DMX universes
            while strip_pixels_counter > 0:
                # Handle case when not enough pixels to fill up a whole universe
                if strip_pixels_counter < 170:
                    dmx_data.append((universe_count, list(scaled_frame[channel_count: channel_count + strip_pixels_counter * 3])))
                    channel_count += strip_pixels_counter * 3
                    strip_pixels_counter -= strip_pixels_counter
                else:
                    dmx_data.append((universe_count, list(scaled_frame[channel_count: channel_count + 170 * 3])))
                    channel_count += 170 * 3  # we arranged 170 pixels which use 510 channels
                    strip_pixels_counter -= 170
                    
                # When a universe is filled, increment the universe count
                if channel_count % 510 == 0:
                    universe_count += 1  
                    
        #  logging.debug(dmx_data)
        return dmx_data


    #  def convert_frame_to_sacn_data(self, frame: np.array) -> List[List[int]]:
    #      np_frame = np.frombuffer(frame, dtype=np.uint8)
    #      scaled_brightness = self.brightness / 255
    #      scaled_frame = (np_frame * scaled_brightness).astype(np.uint8)
    #      scaled_frame = scaled_frame.tobytes()
    #
    #      dmx_data = []  # List to store the DMX data
    #      universe_count = 1  # Variable to keep track of the universe count
    #      channel_count = 1  # Variable to keep track of the channel count
    #
    #
    #      for strip in range(self.num_strips):
    #          strip_pixels_counter = self.num_pixels # Number of pixels for the current strip
    #
    #          # Keep track of used universes within strip
    #          strip_universe_use_counter = 0
    #          channel_count = 1  # Reset channel count at the start of each strip
    #
    #          # As long as we have pixels to arrange within DMX universes
    #          while strip_pixels_counter > 0:
    #              # Handle case when not enough pixels to fill up a whole universe
    #              if strip_pixels_counter < 170:
    #                  dmx_data.append((universe_count, list(scaled_frame[channel_count - 1: channel_count - 1 + strip_pixels_counter * 3])))
    #                  channel_count += strip_pixels_counter * 3
    #                  strip_pixels_counter -= strip_pixels_counter
    #              else:
    #                  dmx_data.append((universe_count, list(scaled_frame[channel_count - 1: channel_count - 1 + 170 * 3])))
    #                  channel_count += 170 * 3  # we arranged 170 pixels which use 510 channels
    #                  strip_pixels_counter -= 170
    #
    #              strip_universe_use_counter += 1
    #              universe_count += 1  # move to next universe for either next part of strip or the new strip
    #
    #          #  # Log how much of universes each strip used
    #          #  logging.debug(f'Strip:{strip} used {strip_universe_use_counter} universes,  channel count: {channel_count}')
    #
    #      # as we return pixel color data alongside with channels and universe ids, we need to use a tuple or similar construct
    #      return dmx_data

    def send_sacn_data(self, data: List[Tuple[int, List[int]]]):
        # Transform list of tuples into dictionary
        data_dict = {universe_id: universe_data for universe_id, universe_data in data}
        
        # Go through all senders
        for i in range(self.universe_count):
            universe_id = i + 1
            universe_data = data_dict.get(universe_id)
            
            if universe_data is not None:
                # if data for this universe exists, send it
                self.sender[universe_id].dmx_data = universe_data
                #  if universe_id > 51:
                #      logging.debug(f'Universe {universe_id} data: {universe_data}')
            #  else:
            #      logging.warning(f'No data for universe {universe_id}')
            #  else:
            #      # if no data for this universe, send zeros (off) to all channels
            #      self.sender[universe_id].dmx_data = [0] * 512

    #  def send_sacn_data(self, data: List[List[int]]):
    #      for i in range(len(data)):
    #          self.sender[i+1].dmx_data = data[i]
    #          #  # scale data by brightness
    #          #  scaled_data = [round(byte * float(self.brightness / 255.0)) for byte in data[i]]
    #          #  self.sender[i+1].dmx_data = scaled_data

    def send_frame(self, frame: np.array):
        data = self.convert_frame_to_sacn_data(frame)
        self.send_sacn_data(data)
