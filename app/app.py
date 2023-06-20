from os import name
import time
import zmq
import zmq.asyncio
import asyncio
import logging
import json
#  from modules.video_player import VideoPlayer, VideoPlayerState, VideoPlayerMode
#  from modules.colorlight import ColorLightDisplay
from modules.ww_player import WWVideoPlayer, VideoPlayerState, VideoPlayerMode
from modules.sacn_send import SacnSend
from modules.zmqcomm import socket_connect, listen_to_messages

LAST_MSG_TIME = time.time()

class PlayerApp:
    def __init__(self, config):
        self.config = config

        logging.basicConfig(level=self.get_log_level(config['debug']['log_level']))

        self.ctx = zmq.asyncio.Context()

        # Publish for the apps
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{config['zmq']['ip_bind']}:{config['zmq']['port_player_pub']}")  # Publish to the player app

        # Subscribe to the server app
        self.server_sub_socket = self.ctx.socket(zmq.SUB)
        socket_connect(self.server_sub_socket, config['zmq']['ip_connect'], config['zmq']['port_server_pub'])
        # Subscribe to the serial app
        self.serial_sub_socket = self.ctx.socket(zmq.SUB)
        socket_connect(self.serial_sub_socket, config['zmq']['ip_connect'], config['zmq']['port_serial_pub'])

        self.ws_queue = asyncio.Queue()

        self.command_dict = {
            "play": self.play,
            "pause": self.pause,
            "stop": self.stop,
            "restart": self.restart,
            "prev": self.prev,
            "next": self.next,
            "get_state": self.get_state,
            "set_brightness": self.set_brightness,
            "get_brightness": self.get_brightness,
            "repeat": self.repeat,
            "repeat_one": self.repeat_one,
            "repeat_none": self.repeat_none,
            #  "get_mode": self.get_mode,
            "set_fps": self.set_fps,
            "get_fps": self.get_fps,
            "get_current_media": self.get_current_video,
            #  "set_playlist": self.set_playlist
        }


        #handle dummy config setting
        self.dummy_key_s = self.config['debug']['dummy_send']
        self.dummy_key = False
        if self.dummy_key_s == "True":
            self.dummy_key = True

        #  self.display = ColorLightDisplay(
        #      interface=config['interface'],
        #      brightness_level=config['brightness_level'],
        #      dummy= dummy_key
        #  )
        #
        #  self.video_player = VideoPlayer(self.ws_queue, config['video_dir'], display_callback=self.display.display_frame)

        logging.debug("sacn address: " + config['sacn']['bind_address'])

        if not self.dummy_key:
            self.sacn = SacnSend(config['sacn']['bind_address'], dummy=self.dummy_key, brightness=config['brightness_level'], multicast = config['sacn']['multicast'] == 1 , universe_count=config['sacn']['universe_count'])

        self.video_player = WWVideoPlayer(self.ws_queue, video_dir=config['video_dir'], display_callback=self.sacn.send_frame)

        logging.debug("Player app initialized")
        self.video_player.play()

    def get_log_level(self, level):
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(level.upper(), logging.INFO)



###############################################################
###############################################################

    async def play(self, params):
        self.video_player.play()

    async def pause(self, params):
        self.video_player.pause()

    async def stop(self, params):
        self.video_player.stop()

    async def restart(self, params):
        logging.debug("Received restart")
        self.video_player.restart_video()

    async def prev(self, params):
        logging.debug("Received prev")
        self.video_player.prev_video()

    async def next(self, params):
        logging.debug("Received next")
        self.video_player.next_video()

    async def get_state(self, params):
        state = "playing" if self.video_player.state == VideoPlayerState.PLAYING else "paused"
        if self.video_player.state == VideoPlayerState.STOPPED:
            state = "stopped"
            logging.debug("Received get_state: " + state)
        #  await self.sock.send_string(str(state))

    async def set_brightness(self, params):
        params = params.split(' ')
        brightness = int(float(params[1])) if params else None
        if brightness is not None:
            #  self.display.brightness_level = int(brightness)
            self.sacn.brightness= int(brightness)

    async def get_brightness(self, params):
        #  await self.sock.send_string(str(self.sacn.brightness))
        pass

    async def set_fps(self, params):
        params = params.split(' ')
        fps = int(float(params[1])) if params else None
        if fps is not None:
            self.video_player.fps = fps
    
    async def get_fps(self, params):
        #  await self.sock.send_string(str(self.video_player.fps))
        pass

    async def repeat(self, params):
        logging.debug("Received repeat")
        self.video_player.mode = VideoPlayerMode.REPEAT

    async def repeat_one(self, params):
        logging.debug("Received repeat_one")
        self.video_player.mode = VideoPlayerMode.REPEAT_ONE

    async def repeat_none(self, params):
        logging.debug("Received repeat_none")
        self.video_player.mode = VideoPlayerMode.REPEAT_NONE

    async def get_current_video(self, params):
        #  #return current video name
        #  await self.sock.send_string(self.video_player.get_current_video_name())
        #  #  await self.sock.send_string(self.video_player.current_video)
        pass

###############################################################
###############################################################

    async def pubUpdate(self):
        logging.info("Starting pubUpdate")
        while True:
            try:
                # send player state
                if self.sacn.brightness:
                    await self.pub_socket.send_string("brightness "+str(self.sacn.brightness))
                if self.video_player.fps:
                    await self.pub_socket.send_string("fps "+str(self.video_player.fps))
                if self.video_player.state:
                    await self.pub_socket.send_string("state "+str(self.video_player.state))
                if self.video_player.mode:
                    await self.pub_socket.send_string("mode "+str(self.video_player.mode))
                if self.video_player.current_video:
                    await self.pub_socket.send_string("current_media "+str(self.video_player.current_video))
                #  pub_socket.send_string("")

            except zmq.ZMQError as zmq_error:
                logging.error(f"ZMQ Error occurred: {str(zmq_error)}")
        
            except Exception as e:
                logging.error(f"A zmq run error occurred: {str(e)}")
            await asyncio.sleep(0.1)
                #  await self.pub_socket.send_string(f"An error occurred: {str(e)}")


    async def process_message(self, sock, message):
        logging.debug(f"Received message: {message}")
        try:
            command = message.split(' ', 1)[0]
            logging.debug(f"Received command: {command}")

            if command in self.command_dict:  # check if command exists in command_dict
                await self.command_dict[command](message)
            else:
                await sock.send_string("Unknown command")
                logging.warning(f"Unknown command received: {command}")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")


    async def run(self):

            #  socket_connect(self.server_sub_socket, config['zmq']['ip_connect'], config['zmq']['port_server_pub']),
            #  socket_connect(self.serial_sub_socket, config['zmq']['ip_connect'], config['zmq']['port_serial_pub']),
        await asyncio.gather(
            listen_to_messages(self.server_sub_socket, self.process_message),
            listen_to_messages(self.serial_sub_socket, self.process_message),
            self.pubUpdate()
        )
        logging.info("Async tasks created")


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    config = load_config('config/config.json')
    app = PlayerApp(config)
    asyncio.run(app.run())

