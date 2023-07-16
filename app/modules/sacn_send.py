import sacn
import atexit
import array
import logging
import numpy as np
import math 
import time
import itertools
import cProfile


from typing import Callable, Optional, List, Dict, Union, Tuple

class SacnSend:
    def __init__(self, bind_address = "127.0.0.1", num_strips=1, num_pixels=1, multicast=True, dummy=False, brightness = 50.0):
        self.multi = multicast
        self.sender = sacn.sACNsender(bind_address)
        self.sender.fps = 60
        self.brightness = brightness
        self.set_brightness(brightness)
        self.num_strips = num_strips
        self.num_pixels = num_pixels
        self.universe_count = math.ceil( (num_strips * num_pixels * 3) / 510)

        self.chunk_template = [0] * 512
        self.chunk_template[0] = 0  

        logging.info(f"sACN Number of pixels: {self.num_pixels}, Number of strips: {self.num_strips}, Universe count: {self.universe_count}")

        for i in range(1, self.universe_count + 1):
            if not dummy:
                self.sender.activate_output(i) 
            if self.multi:
                self.sender[i].multicast = True

        self.sender.start()
        atexit.register(self.sender.stop)

    def set_brightness(self, brightness):
        self.brightness = brightness
        self.brightness_table = bytearray(int(i * brightness / 255) for i in range(256))
        
    def profile_convert_frame_to_sacn_data(self, frame):
        profiler = cProfile.Profile()
        profiler.enable()

        data = self.convert_frame_to_sacn_data(frame)

        profiler.disable()
        profiler.print_stats(sort='time')

        return data

    def convert_frame_to_sacn_data(self, frame):
        np_frame = (np.frombuffer(frame, dtype=np.uint8) * self.brightness) // 255
        #  # Convert to numpy array if frame is not already one
        #  if not isinstance(frame, np.ndarray):
        #      np_frame = np.array(frame, dtype=np.uint8)
        #
        #  # Apply brightness scaling
        #  np_frame = (np_frame * (self.brightness / 255)).astype(np.uint8, casting='unsafe')

        # Split into groups of 510
        packetized_frame = np.split(np_frame, np.arange(510, len(np_frame), 510))

        # Convert numpy arrays directly to lists
        packetized_frame = [packet.tolist() for packet in packetized_frame]

        return packetized_frame


    #  def convert_frame_to_sacn_data(self, frame):
    #      #  np_frame = np.frombuffer(frame, dtype=np.uint8)
    #      #  np_frame = (np_frame * (self.brightness / 255)).astype('uint8', casting='unsafe')
    #
    #      #  np_frame = (np.frombuffer(frame, dtype=np.uint8).astype(float) * (self.brightness / 255)).astype('uint8')
    #      # Directly convert to uint8 after scaling
    #      np_frame = (np.frombuffer(frame, dtype=np.uint8) * self.brightness) // 255
    #
    #      # Concatenate original data with padding in one step
    #      padded_frame = np.pad(np_frame, (0, 512 - len(np_frame)), 'constant', constant_values=0)
    #
    #      # Split into groups of 510 (sACN packets can only have 510 channels of DMX data)
    #      packetized_frame = np.split(padded_frame, np.ceil(len(padded_frame) / 510))
    #      #  packetized_frame = np.split(padded_frame, np.ceil(len(padded_frame) / 512))
    #
    #      # Convert numpy arrays directly to lists
    #      packetized_frame = [packet.tolist() for packet in packetized_frame]
    #
    #      return packetized_frame


    ################################
    ## WERKS, SLOW
    ################################
    #  def convert_frame_to_sacn_data(self, frame):
    #      start_time = time.time()
    #      np_frame = np.frombuffer(frame, dtype=np.uint8)
    #
    #      # Scale brightness
    #      np_frame = (np_frame * (self.brightness / 255)).astype(np.uint8)
    #
    #      end_time = time.time()
    #      #  logging.debug(f'Conversion and scaling time: {end_time - start_time}')
    #
    #      # Reshape to (-1, 510), padding with zeros if necessary
    #      start_time = time.time()
    #      total_size = np_frame.size
    #      num_chunks = (total_size + 511) // 512  # Compute number of chunks, rounding up
    #      padded_frame = np.zeros(num_chunks * 512, dtype=np.uint8)
    #      padded_frame[:total_size] = np_frame  # Copy original data
    #
    #      # Now reshape to (-1, 510)
    #      chunks = padded_frame.reshape(-1, 512)
    #      #  # Reshape to (-1, 510), padding with zeros if necessary
    #      #  start_time = time.time()
    #      #  total_size = np_frame.size
    #      #  num_chunks = (total_size + 509) // 510  # Compute number of chunks, rounding up
    #      #  padded_frame = np.zeros(num_chunks * 510, dtype=np.uint8)
    #      #  padded_frame[:total_size] = np_frame  # Copy original data
    #      #
    #      #  # Now reshape to (-1, 510)
    #      #  chunks = padded_frame.reshape(-1, 510)
    #
    #      end_time = time.time()
    #      #  logging.debug(f'Chunking time: {end_time - start_time}')
    #
    #      return chunks


    ##################### 22 FPS
    #  def convert_frame_to_sacn_data(self, frame: np.array) -> List[List[int]]:
    #      start_time = time.time()
    #
    #      np_frame = np.frombuffer(frame, dtype=np.uint8)
    #      conversion_time = time.time()
    #      np_frame = (np_frame * (self.brightness / 255)).astype(np.uint8)  # Scale and convert to uint8
    #      scaling_time = time.time()
    #      flattened_frame = np_frame.flatten()
    #
    #      # Initialize the universe count
    #      universe_count = 1
    #
    #      # Initialize an empty list to hold the DMX data
    #      dmx_data = []
    #
    #      # Prepare a template with padding in advance
    #      chunk_template = np.full(512, fill_value=0xFF, dtype=np.uint8)
    #      chunk_template[0] = 0x00  # start code
    #
    #      # Iterate over the flattened frame in chunks of 510
    #      for i in range(0, len(flattened_frame), 510):
    #          chunk = flattened_frame[i:i+510]
    #          chunk_template[1:1+len(chunk)] = chunk  # Assign to a slice that matches the size of the chunk
    #
    #          # Add the chunk to dmx_data along with the current universe count
    #          dmx_data.append((universe_count, bytes(chunk_template)))
    #
    #          # Reset chunk_template for next iteration
    #          chunk_template[1:1+len(chunk)] = 0xFF
    #
    #          # Increment the universe count
    #          universe_count += 1
    #
    #      chunking_time = time.time()
    #
    #      #  # Number of channels per strip
    #      #  channels_per_strip = self.num_pixels * 3
    #      #
    #      #  # Log the universe, channel, and data for the first pixel of each strip
    #      #  for strip in range(self.num_strips):
    #      #      first_pixel_index = strip * channels_per_strip
    #      #      universe = first_pixel_index // 510 + 1
    #      #      channel = first_pixel_index % 510 + 1
    #      #      data = flattened_frame[first_pixel_index:first_pixel_index+3]
    #      #      logging.debug(f"strip {strip} pixel 0 is universe {universe}, channel {channel}, data {data}")
    #
    #      #  logging_time = time.time()
    #      #
    #      #  logging.info(f"Conversion time: {conversion_time - start_time}")
    #      #  logging.info(f"Scaling time: {scaling_time - conversion_time}")
    #      #  logging.info(f"Chunking time: {chunking_time - scaling_time}")
    #      #  logging.info(f"Logging time: {logging_time - chunking_time}")
    #
    #      return dmx_data



    #####################################
    # OG convert_frame_to_sacn_data
    #####################################
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
    def send_sacn_data(self, data):

      for universe, packet in enumerate(data, start=1):
          self.universe = universe
          self.sender[universe].dmx_data = packet
          #  self.sender.flush()

        #  for universe_id, universe_data in enumerate(data, start=1):  # starts numbering from 1
        #      self.sender[universe_id].dmx_data = universe_data.tolist()



    #  def send_sacn_data(self, data: List[Tuple[int, List[int]]]):
    #      # Transform list of tuples into dictionary
    #      data_dict = {universe_id: universe_data for universe_id, universe_data in data}
    #
    #      # Go through all senders
    #      for i in range(self.universe_count):
    #          universe_id = i + 1
    #          universe_data = data_dict.get(universe_id)
    #
    #          if universe_data is not None:
    #              # if data for this universe exists, send it
    #              self.sender[universe_id].dmx_data = universe_data
    #              #  if universe_id > 51:
    #              #      logging.debug(f'Universe {universe_id} data: {universe_data}')
    #          #  else:
    #          #      logging.warning(f'No data for universe {universe_id}')
    #          #  else:
    #          #      # if no data for this universe, send zeros (off) to all channels
    #          #      self.sender[universe_id].dmx_data = [0] * 512

    #  def send_sacn_data(self, data: List[List[int]]):
    #      for i in range(len(data)):
    #          self.sender[i+1].dmx_data = data[i]
    #          #  # scale data by brightness
    #          #  scaled_data = [round(byte * float(self.brightness / 255.0)) for byte in data[i]]
    #          #  self.sender[i+1].dmx_data = scaled_data

    def send_frame(self, frame: np.array):
        pr = cProfile.Profile()
        pr.enable()

        data = self.convert_frame_to_sacn_data(frame)
        #  data = self.profile_convert_frame_to_sacn_data(frame)
        self.send_sacn_data(data)

        pr.disable()
        pr.print_stats(sort='time')

