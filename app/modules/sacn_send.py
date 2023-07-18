import sacn
import atexit
import array
import logging
import numpy as np
import math 
import time
import itertools
import cProfile
import uuid


import socket
import struct

class SacnSend:
    def __init__(self, bind_address = "127.0.0.1", num_strips=1, num_pixels=1, multicast=True, dummy=False, brightness = 50.0, start_universe=1, port=5568):
        self.bind_address = bind_address
        self.multi = multicast
        self.brightness = brightness
        self.set_brightness(brightness)
        self.num_strips = num_strips
        self.num_pixels = num_pixels
        self.num_universes = math.ceil( (num_strips * num_pixels * 3) / 510)

        self.start_universe = start_universe
        self.port = port
        # Create a UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # Set SO_REUSEADDR to allow multiple instances of the application on the same machine
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind the socket to a specific network interface
        self.socket.bind((self.bind_address, 0))

    def set_brightness(self, brightness):
        self.brightness = brightness

    @staticmethod
    def multicast_ip_generator(universe):
        return f"239.255.{universe >> 8}.{universe & 0xFF}"


    def _create_header(self, universe, source_name="PoleFX", priority=100, sequence_number=0, options=0, address_increment=1, property_value_count=512, cid="polefx"):
        acn_pid = "ASC-E1.17\x00\x00\x00"
        flags_and_length = struct.pack('!H', 0x7000 | (638 & 0x0FFF))
        uuid_device = uuid.uuid5(uuid.NAMESPACE_DNS, cid)
      
        vector = struct.pack('!I', 4)
        source_name = (source_name + '\x00' * 64)[:64]
        priority = struct.pack('!B', priority)
        sync_address = struct.pack('!H', 0)
      
        sequence_number = struct.pack('!B', sequence_number)
        options = struct.pack('!B', options)
        universe = struct.pack('!H', universe)
        framing_flags_and_length = struct.pack('!H', 0x7000 | (638 & 0x0FFF))
        framing_vector = struct.pack('!I', 2)
        dmp_flags_and_length = struct.pack('!H', 0x7000 | (513 & 0x0FFF))
        dmp_vector = struct.pack('!B', 2)
        address_and_data_type = struct.pack('!B', 0xa1)
        first_property_address = struct.pack('!H', 0)
        address_increment = struct.pack('!H', address_increment)
        property_value_count = struct.pack('!H', property_value_count)

        header = struct.pack('!16sHH16sI64sBHHBBH16sHHIHHBBHBBHH', 
                             acn_pid.encode(), 
                             0x7000 | (638 & 0x0FFF),  
                             uuid_device.bytes, 
                             4,  
                             source_name.encode(), 
                             100,  
                             0,  # Reserved
                             0,  
                             0,  
                             universe,  
                             0x7000 | (638 & 0x0FFF),  
                             2,  
                             source_name.encode(), 
                             100,  
                             0,  
                             0,  # Reserved
                             sequence_number,  
                             options,  
                             universe, 
                             0x7000 | (513 & 0x0FFF),
                             2,
                             0xa1,
                             0,  
                             1,  
                             512)
        return header

    def send_frame(self, frame: np.array):
        # Initialize parameters
        source_name = "litPi"
        cid = "litpi"
        priority = 100
        sequence_number = 0
        options = 0
        address_increment = 1
        property_value_count = 512

        # Convert frame to SACN data
        data = self.convert_frame_to_sacn_data(frame)

        # Send each SACN universe data
        for universe, packet in enumerate(data, start=1):
            # Create header for each frame
            logging.debug(f"universe {universe}")
            header = self._create_header(int(universe), 
                                         source_name=source_name, 
                                         priority=priority, 
                                         sequence_number=sequence_number,
                                         options=options, 
                                         address_increment=address_increment,
                                         property_value_count=property_value_count,
                                         cid=cid)
            # Prepare packet
            packet = header + packet

            # Send packet
            self.socket.sendto(packet, ('239.255.0.' + str(universe), 5568))

            # Increment sequence_number for the next packet
            sequence_number = sequence_number + 1 if sequence_number < 255 else 0


    def convert_frame_to_sacn_data(self, frame):
        # Convert frame to numpy array and apply brightness scaling
        np_frame = np.frombuffer(frame, dtype=np.uint8)

        # Calculate the number of chunks and the size of the padded frame
        total_size = np_frame.size
        num_chunks = (total_size + 510) // 511  # Compute number of chunks, rounding up

        # Create a zero-padded frame
        padded_frame = np.zeros(num_chunks * 511, dtype=np.uint8)
        padded_frame[:total_size] = np_frame  # Copy original data

        # Reshape to (-1, 511)
        packetized_frame = padded_frame.reshape(-1, 511)

        # Convert numpy arrays directly to lists of bytes
        packetized_frame = [packet.tobytes() for packet in packetized_frame]

        return packetized_frame

   
    #  def _create_header(self, length, universe):
    #      source_name = 'litPi-sACN'
    #      padded_source_name = (source_name + ('\x00' * (64 - len(source_name)))).encode()
    #      return struct.pack('!16sH64sBxxHBBHxxH',
    #          b'ASC-E1.17\x00\x00\x00',  # preamble
    #          0x7000 | (638 & 0x0FFF),  # flags and length
    #          padded_source_name,  # source name
    #          100,  # priority
    #          0,  # sequence number
    #          0,  # options
    #          universe,  # universe
    #          0x7000 | (length & 0x0FFF),  # flags and length
    #          length  # property value count
    #      )


    #  def _create_header(self, dlen, universe):
    #      return struct.pack('!16sHHHHHHHHIxxHBBBBBBBxxBBBH',
    #                         b'ASC-E1.17\0\0\0',
    #                         0x7000 | (638 + dlen),
    #                         0x7000 | 22,
    #                         0x0000,
    #                         0x0004,
    #                         0x7000 | 38,
    #                         0x0000,
    #                         0x0400,
    #                         0x7000 | 638,
    #                         1,
    #                         0x0000,
    #                         universe,
    #                         0x0201,
    #                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    #                         0x04,
    #                         0x01,
    #                         0x7000 | dlen,
    #                         0x00)

    #
    #  def send_frame(self, frame: np.array):
    #      data_list = self.convert_frame_to_sacn_data(frame)
    #
    #      for universe, data in enumerate(data_list, start=self.start_universe):
    #          multicast_ip = self.multicast_ip_generator(universe)
    #          header = self._create_header(len(data), universe)
    #          packet = header + data
    #          self.sock.sendto(packet, (multicast_ip, self.port))

    #  @staticmethod
    #  def convert_frame_to_sacn_data(frame, brightness=1.0):
    #      # Convert frame to numpy array and apply brightness scaling
    #      np_frame = (np.frombuffer(frame, dtype=np.uint8) * brightness) // 255
    #
    #      # Calculate the number of chunks and the size of the padded frame
    #      total_size = np_frame.size
    #      num_chunks = (total_size + 511) // 512  # Compute number of chunks, rounding up
    #
    #      # Create a zero-padded frame
    #      padded_frame = np.zeros(num_chunks * 512, dtype=np.uint8)
    #      padded_frame[:total_size] = np_frame  # Copy original data
    #
    #      # Reshape to (-1, 512)
    #      packetized_frame = padded_frame.reshape(-1, 512)
    #
    #      # Convert numpy arrays directly to lists of bytes
    #      packetized_frame = [packet.tobytes() for packet in packetized_frame]
    #
    #      return packetized_frame

#
#  from typing import Callable, Optional, List, Dict, Union, Tuple
#
#  class SacnSend:
#      def __init__(self, bind_address = "127.0.0.1", num_strips=1, num_pixels=1, multicast=True, dummy=False, brightness = 50.0):
#          self.multi = multicast
#          self.sender = sacn.sACNsender(bind_address)
#          self.sender.fps = 30
#          self.brightness = brightness
#          self.set_brightness(brightness)
#          self.num_strips = num_strips
#          self.num_pixels = num_pixels
#          self.universe_count = math.ceil( (num_strips * num_pixels * 3) / 510)
#
#          self.chunk_template = [0] * 512
#          self.chunk_template[0] = 0
#
#          logging.info(f"sACN Number of pixels: {self.num_pixels}, Number of strips: {self.num_strips}, Universe count: {self.universe_count}")
#
#          for i in range(1, self.universe_count + 1):
#              if not dummy:
#                  self.sender.activate_output(i)
#              if self.multi:
#                  self.sender[i].multicast = True
#          self.dummy_frame = self.generate_dummy_frame(self.num_strips * self.num_pixels * 3)
#          self.dmx_data = self.generate_dummy_frame(self.num_strips * self.num_pixels * 3)
#
#          self.sender.start()
#          atexit.register(self.sender.stop)
#
#      def generate_dummy_frame_from_real_data(self, real_frame):
#          # Convert real frame to sACN data
#          real_data = self.convert_frame_to_sacn_data(real_frame)
#
#          # Duplicate real_data to create dummy frame
#          dummy_frame = real_data * (self.num_strips * self.num_pixels // len(real_data))
#
#          return dummy_frame
#
#      def generate_dummy_frame(self, total_size):
#          # Create a frame filled with random data
#          np_frame = np.random.randint(low=0, high=256, size=total_size, dtype=np.uint8)
#
#          # Calculate the number of chunks and the size of the padded frame
#          num_chunks = (total_size + 511) // 512  # Compute number of chunks, rounding up
#
#          # Create a zero-padded frame
#          padded_frame = np.zeros(num_chunks * 512, dtype=np.uint8)
#          padded_frame[:total_size] = np_frame  # Copy original data
#
#          # Reshape to (-1, 512)
#          packetized_frame = padded_frame.reshape(-1, 512)
#
#          # Convert numpy arrays directly to lists of bytes
#          packetized_frame = [packet.tobytes() for packet in packetized_frame]
#
#          return packetized_frame
#
#      def set_brightness(self, brightness):
#          self.brightness = brightness
#          self.brightness_table = bytearray(int(i * brightness / 255) for i in range(256))
#
#      def profile_convert_frame_to_sacn_data(self, frame):
#          profiler = cProfile.Profile()
#          profiler.enable()
#
#          data = self.convert_frame_to_sacn_data(frame)
#
#          profiler.disable()
#          profiler.print_stats(sort='time')
#
#          return data
#
#
#      def convert_frame_to_sacn_data(self, frame):
#          # Convert frame to numpy array and apply brightness scaling
#          #  np_frame = (np.frombuffer(frame, dtype=np.uint8) * self.brightness) // 255
#          np_frame = (np.frombuffer(frame, dtype=np.uint8) )
#
#          # Calculate the number of chunks and the size of the padded frame
#          total_size = np_frame.size
#          num_chunks = (total_size + 511) // 512  # Compute number of chunks, rounding up
#
#          # Create a zero-padded frame
#          padded_frame = np.zeros(num_chunks * 512, dtype=np.uint8)
#          padded_frame[:total_size] = np_frame  # Copy original data
#
#          # Reshape to (-1, 512)
#          packetized_frame = padded_frame.reshape(-1, 512)
#
#          # Convert numpy arrays directly to lists of bytes
#          packetized_frame = [packet.tobytes() for packet in packetized_frame]
#
#          return packetized_frame
#
#
#      ################################
#      ## WERKS, SLOW
#      ################################
#      #  def convert_frame_to_sacn_data(self, frame):
#      #      start_time = time.time()
#      #      np_frame = np.frombuffer(frame, dtype=np.uint8)
#      #
#      #      # Scale brightness
#      #      np_frame = (np_frame * (self.brightness / 255)).astype(np.uint8)
#      #
#      #      end_time = time.time()
#      #      #  logging.debug(f'Conversion and scaling time: {end_time - start_time}')
#      #
#      #      # Reshape to (-1, 510), padding with zeros if necessary
#      #      start_time = time.time()
#      #      total_size = np_frame.size
#      #      num_chunks = (total_size + 511) // 512  # Compute number of chunks, rounding up
#      #      padded_frame = np.zeros(num_chunks * 512, dtype=np.uint8)
#      #      padded_frame[:total_size] = np_frame  # Copy original data
#      #
#      #      # Now reshape to (-1, 510)
#      #      chunks = padded_frame.reshape(-1, 512)
#      #      #  # Reshape to (-1, 510), padding with zeros if necessary
#      #      #  start_time = time.time()
#      #      #  total_size = np_frame.size
#      #      #  num_chunks = (total_size + 509) // 510  # Compute number of chunks, rounding up
#      #      #  padded_frame = np.zeros(num_chunks * 510, dtype=np.uint8)
#      #      #  padded_frame[:total_size] = np_frame  # Copy original data
#      #      #
#      #      #  # Now reshape to (-1, 510)
#      #      #  chunks = padded_frame.reshape(-1, 510)
#      #
#      #      end_time = time.time()
#      #      #  logging.debug(f'Chunking time: {end_time - start_time}')
#      #
#      #      return chunks
#
#
#      ##################### 22 FPS
#      #  def convert_frame_to_sacn_data(self, frame: np.array) -> List[List[int]]:
#      #      start_time = time.time()
#      #
#      #      np_frame = np.frombuffer(frame, dtype=np.uint8)
#      #      conversion_time = time.time()
#      #      np_frame = (np_frame * (self.brightness / 255)).astype(np.uint8)  # Scale and convert to uint8
#      #      scaling_time = time.time()
#      #      flattened_frame = np_frame.flatten()
#      #
#      #      # Initialize the universe count
#      #      universe_count = 1
#      #
#      #      # Initialize an empty list to hold the DMX data
#      #      dmx_data = []
#      #
#      #      # Prepare a template with padding in advance
#      #      chunk_template = np.full(512, fill_value=0xFF, dtype=np.uint8)
#      #      chunk_template[0] = 0x00  # start code
#      #
#      #      # Iterate over the flattened frame in chunks of 510
#      #      for i in range(0, len(flattened_frame), 510):
#      #          chunk = flattened_frame[i:i+510]
#      #          chunk_template[1:1+len(chunk)] = chunk  # Assign to a slice that matches the size of the chunk
#      #
#      #          # Add the chunk to dmx_data along with the current universe count
#      #          dmx_data.append((universe_count, bytes(chunk_template)))
#      #
#      #          # Reset chunk_template for next iteration
#      #          chunk_template[1:1+len(chunk)] = 0xFF
#      #
#      #          # Increment the universe count
#      #          universe_count += 1
#      #
#      #      chunking_time = time.time()
#      #
#      #      #  # Number of channels per strip
#      #      #  channels_per_strip = self.num_pixels * 3
#      #      #
#      #      #  # Log the universe, channel, and data for the first pixel of each strip
#      #      #  for strip in range(self.num_strips):
#      #      #      first_pixel_index = strip * channels_per_strip
#      #      #      universe = first_pixel_index // 510 + 1
#      #      #      channel = first_pixel_index % 510 + 1
#      #      #      data = flattened_frame[first_pixel_index:first_pixel_index+3]
#      #      #      logging.debug(f"strip {strip} pixel 0 is universe {universe}, channel {channel}, data {data}")
#      #
#      #      #  logging_time = time.time()
#      #      #
#      #      #  logging.info(f"Conversion time: {conversion_time - start_time}")
#      #      #  logging.info(f"Scaling time: {scaling_time - conversion_time}")
#      #      #  logging.info(f"Chunking time: {chunking_time - scaling_time}")
#      #      #  logging.info(f"Logging time: {logging_time - chunking_time}")
#      #
#      #      return dmx_data
#
#
#
#      #####################################
#      # OG convert_frame_to_sacn_data
#      #####################################
#      #  def convert_frame_to_sacn_data(self, frame: np.array) -> List[List[int]]:
#      #      np_frame = np.frombuffer(frame, dtype=np.uint8)
#      #      scaled_brightness = self.brightness / 255
#      #      scaled_frame = (np_frame * scaled_brightness).astype(np.uint8)
#      #      scaled_frame = scaled_frame.tobytes()
#      #
#      #      dmx_data = []  # List to store the DMX data
#      #      universe_count = 1  # Variable to keep track of the universe count
#      #      channel_count = 1  # Variable to keep track of the channel count
#      #
#      #
#      #      for strip in range(self.num_strips):
#      #          strip_pixels_counter = self.num_pixels # Number of pixels for the current strip
#      #
#      #          # Keep track of used universes within strip
#      #          strip_universe_use_counter = 0
#      #          channel_count = 1  # Reset channel count at the start of each strip
#      #
#      #          # As long as we have pixels to arrange within DMX universes
#      #          while strip_pixels_counter > 0:
#      #              # Handle case when not enough pixels to fill up a whole universe
#      #              if strip_pixels_counter < 170:
#      #                  dmx_data.append((universe_count, list(scaled_frame[channel_count - 1: channel_count - 1 + strip_pixels_counter * 3])))
#      #                  channel_count += strip_pixels_counter * 3
#      #                  strip_pixels_counter -= strip_pixels_counter
#      #              else:
#      #                  dmx_data.append((universe_count, list(scaled_frame[channel_count - 1: channel_count - 1 + 170 * 3])))
#      #                  channel_count += 170 * 3  # we arranged 170 pixels which use 510 channels
#      #                  strip_pixels_counter -= 170
#      #
#      #              strip_universe_use_counter += 1
#      #              universe_count += 1  # move to next universe for either next part of strip or the new strip
#      #
#      #          #  # Log how much of universes each strip used
#      #          #  logging.debug(f'Strip:{strip} used {strip_universe_use_counter} universes,  channel count: {channel_count}')
#      #
#      #      # as we return pixel color data alongside with channels and universe ids, we need to use a tuple or similar construct
#      #      return dmx_data
#      def send_sacn_data(self, data):
#
#          #  self.sender.manual_flush = True
#          for universe, packet in enumerate(data, start=1):
#              self.sender[universe].dmx_data = packet
#          #  self.sender.flush()
#
#        #  for universe, packet in enumerate(data, start=1):
#        #      self.universe = universe
#        #      self.sender[universe].dmx_data = packet
#        #      #  self.sender.flush()
#        #
#        #    #  for universe_id, universe_data in enumerate(data, start=1):  # starts numbering from 1
#        #    #      self.sender[universe_id].dmx_data = universe_data.tolist()
#
#
#
#      #  def send_sacn_data(self, data: List[Tuple[int, List[int]]]):
#      #      # Transform list of tuples into dictionary
#      #      data_dict = {universe_id: universe_data for universe_id, universe_data in data}
#      #
#      #      # Go through all senders
#      #      for i in range(self.universe_count):
#      #          universe_id = i + 1
#      #          universe_data = data_dict.get(universe_id)
#      #
#      #          if universe_data is not None:
#      #              # if data for this universe exists, send it
#      #              self.sender[universe_id].dmx_data = universe_data
#      #              #  if universe_id > 51:
#      #              #      logging.debug(f'Universe {universe_id} data: {universe_data}')
#      #          #  else:
#      #          #      logging.warning(f'No data for universe {universe_id}')
#      #          #  else:
#      #          #      # if no data for this universe, send zeros (off) to all channels
#      #          #      self.sender[universe_id].dmx_data = [0] * 512
#
#      #  def send_sacn_data(self, data: List[List[int]]):
#      #      for i in range(len(data)):
#      #          self.sender[i+1].dmx_data = data[i]
#      #          #  # scale data by brightness
#      #          #  scaled_data = [round(byte * float(self.brightness / 255.0)) for byte in data[i]]
#      #          #  self.sender[i+1].dmx_data = scaled_data
#
#      def compare_data(self, real_data, dummy_data):
#          #  logging.debug("Real data sample:", real_data[:10])  # Print first 10 elements
#          #  logging.debug("Dummy data sample:", dummy_data[:10])  # Print first 10 elements
#
#          logging.debug(f"Real data type:  {type(real_data) }")
#          logging.debug(f"Dummy data type: {type(dummy_data)}")
#
#          logging.debug(f"Real data len:, {len(real_data) }")
#          logging.debug(f"Dummy data len: {len(dummy_data)}")
#
#          for i in range(0, len(dummy_data)):
#            logging.debug(f"Dummy data {i} len: {len(dummy_data[i])}")
#          for i in range(0, len(real_data)):
#            logging.debug(f"real data {i} len: {len(real_data[i])}")
#
#          #  # Compare the first few elements of each list
#          #  for i in range(min(10, len(real_data), len(dummy_data))):
#          #        logging.debug(f"Element {i} - Real: {real_data[i]}, Dummy: {dummy_data[i]}")
#
#
#          #  logging.debug("Real data size:", len(real_data))
#          #  logging.debug("Dummy data size:", len(dummy_data))
#
#      def send_frame(self, frame: np.array):
#          #  pr = cProfile.Profile()
#          #  pr.enable()
#
#          self.dmx_data = self.convert_frame_to_sacn_data(frame)
#          #  data = self.profile_convert_frame_to_sacn_data(frame)
#          #create dummy data
#          self.send_sacn_data(self.dmx_data)
#  # Generate dummy frame from real data
#          #  dummy_frame = self.generate_dummy_frame_from_real_data(frame)
#
#          # Send dummy frame
#          #  self.send_sacn_data(dummy_fame)
#          #  self.dummy_frame = self.generate_dummy_frame(self.num_strips * self.num_pixels* 3)
#          #  self.compare_data(self.dmx_data, self.dummy_frame)
#          #  self.send_sacn_data(self.dummy_frame)
#
#          #  pr.disable()
#          #  pr.print_stats(sort='time')
#
