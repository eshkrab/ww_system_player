import os
import cv2
import glob
import time
import array
import json
import threading
import logging
import socket
import atexit
import numpy as np
from enum import Enum
from collections import deque
from typing import Callable, Optional, List, Dict, Union

import sacn

from modules.ww_utils import WWFile

class VideoPlayerState(Enum):
    PLAYING = 1
    STOPPED = 2
    PAUSED = 3

class VideoPlayerMode(Enum):
    REPEAT = 1
    REPEAT_ONE = 2
    REPEAT_NONE = 3

class WWVideoPlayer:
    def __init__(self, ws_queue, video_dir: str, fps: int = 30,
                 display_callback: Optional[Callable[[np.array], None]] = None):
        self.video_dir = video_dir
        self.ws_queue = ws_queue
        self.fps = fps
        self.state = VideoPlayerState.STOPPED
        self.mode = VideoPlayerMode.REPEAT
        self.current_video = None
        self.playlist: List[Dict[str, Union[str, VideoPlayerMode]]] = []
        self.playlist_path = os.path.join(video_dir, "playlist.json")
        self.lock = threading.Lock()
        self.current_video_index = 0
        self.display_callback = display_callback
        self.sender = sacn.sACNsender()
        for i in range(1, 30):
            self.sender.activate_output(i)
            self.sender[i].multicast = True
        logging.debug("sACN sender activated")
        self.sender.start()
        atexit.register(self.sender.stop)
        self.stop_event = threading.Event()  
        #  self.playback_thread = None

        self.load_playlist()

    def get_current_video_name(self):
        filepath = self.playlist[self.current_video_index]["filepath"]
        return os.path.basename(filepath)

    def play(self):
        with self.lock:
            if self.state != VideoPlayerState.PLAYING:
                logging.debug("PLAYING")
                self.state = VideoPlayerState.PLAYING
                if not hasattr(self, "playback_thread") or not self.playback_thread.is_alive():
                    self.stop_event.clear()
                    self.playback_thread = threading.Thread(target=self.playback_loop)
                    logging.debug("Starting playback thread")
                    self.playback_thread.start()

    def stop(self):
        logging.debug("Stopping before lock")
        with self.lock:
            if self.state != VideoPlayerState.STOPPED:
                logging.debug("STOPPING")
                self.state = VideoPlayerState.STOPPED
                self.playlist.clear()
                self.current_video = None
                if self.playback_thread and self.playback_thread.is_alive():
                    logging.debug("Stopping playback thread")
                    self.stop_event.set()
                    self.playback_thread.join()
                    self.playback_thread = None
        logging.debug("Stopped after lock")

    def pause(self):
        with self.lock:
            if self.state != VideoPlayerState.PAUSED:
                self.state = VideoPlayerState.PAUSED

    def resume(self):
        with self.lock:
            if self.state == VideoPlayerState.PAUSED:
                self.play()

    def next_video(self):
        with self.lock:
            self.current_video_index = (self.current_video_index + 1) % len(self.playlist)
            self.load_video(self.current_video_index)

    def prev_video(self):
        with self.lock:
            self.current_video_index = (self.current_video_index - 1) % len(self.playlist)
            self.load_video(self.current_video_index)

    def restart_video(self):
        with self.lock:
            self.load_video(self.current_video_index)

    def load_video(self, index):
        playlist = self.playlist["playlist"]
        filepath = playlist[index]["filepath"]
        logging.debug("LOADING VIDEO %s", filepath)
        self.current_video = WWFile(filepath)
        #  self.current_video.start()
        #  if self.display_callback:
        #      self.display_callback(self.current_video)

    def playback_loop(self):
        #  while True:
        while not self.stop_event.is_set():
            with self.lock:
                if self.state == VideoPlayerState.STOPPED:
                    break

                elif self.state == VideoPlayerState.PAUSED:
                    time.sleep(0.01)
                    continue

                elif self.state == VideoPlayerState.PLAYING:
                    if not self.current_video and self.playlist:
                        self.load_video(self.current_video_index)

                    if self.current_video:
                        self.current_video.update()
                        frame = self.current_video.get_next_frame()
                        if frame is not None:
                            logging.debug("Sending frame")
                            sacn_data = self.convert_frame_to_sacn_data(frame)
                            self.send_sacn_data(sacn_data)
                        else:
                            self.current_video = None
                            if self.mode == VideoPlayerMode.REPEAT_ONE:
                                self.restart_video()
                            elif self.mode == VideoPlayerMode.REPEAT:
                                self.next_video()
                            elif self.mode == VideoPlayerMode.REPEAT_NONE:
                                if self.current_video_index < len(self.playlist["playlist"]) - 1:
                                    self.next_video()
                                else:
                                    self.stop()
                                #  self.stop()

            if self.stop_event.wait(1 / self.fps):  # returns immediately if the event is set, else waits for the timeout
                logging.debug("Stop event set, breaking")
                #  pass
                break
            #  time.sleep(1 / self.fps)

    def convert_frame_to_sacn_data(self, frame: np.array) -> List[int]:
        # Convert WW animation frame to sACN data format
        # implementation goes here
        dmx_data = array.array('B')
        for i in range(0, len(frame), 3):
            dmx_data.append(frame[i])
        return dmx_data

    def send_sacn_data(self, data: List[int]):
        #  self.sender[1].dmx_data = array.array('B', data)
        for i in range(1, 30):
            self.sender[i].dmx_data = array.array('B', data)
            #  self.sender.send_dmx(i, data)
            logging.debug("Sending DMX data to universe %d", i)

    def load_playlist(self):
        if os.path.exists(self.playlist_path):
            with open(self.playlist_path, "r") as f:
                self.playlist = json.load(f)
        else:
            self.playlist = [{"filepath": x, "mode": "REPEAT"} for x in glob.glob(self.video_dir + "/*.avi")]
            self.save_playlist()

    def save_playlist(self):
        with open(self.playlist_path, "w") as f:
            json.dump(self.playlist, f)

