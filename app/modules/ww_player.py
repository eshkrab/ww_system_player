import os
import cv2
import glob
import time
import array
import json
import threading
import logging
import socket

import numpy as np
from enum import Enum
from collections import deque
from typing import Callable, Optional, List, Dict, Union


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

    def __init__(self, ws_queue, video_dir: str, fps: int = 30, display_callback: Optional[Callable[[np.array], None]] = None):
        self.video_dir = video_dir
        self.ws_queue = ws_queue
        self.fps = fps
        self.state = VideoPlayerState.STOPPED
        self.mode = VideoPlayerMode.REPEAT
        self.current_video = None
        self.playlist: List[Dict[str, Union[str, VideoPlayerMode]]] = []
        self.playlist_path = os.path.join(video_dir, "playlist.json")
        self.fade_factor = 1.0
        self.fade_speed = 0.01

        self.lock = threading.Lock()
        self.current_video_index = 0
        self.display_callback = display_callback

        self.playback_thread = threading.Thread(target=lambda: None)
        self.playback_thread.start()
        self.playback_thread.join()

        self.stop_event = threading.Event()

        self.load_playlist()

    def get_current_video_name(self):
        filepath = self.playlist[self.current_video_index]["filepath"]
        return os.path.basename(filepath)

    def play(self):
        with self.lock:
            if self.state != VideoPlayerState.PLAYING:
                logging.debug("PLAYING")
                self.state = VideoPlayerState.PLAYING
                if not self.playback_thread.is_alive():
                    self.stop_event.clear()
                    self.playback_thread = threading.Thread(target=self.playback_loop)
                    logging.debug("Starting playback thread")
                    self.playback_thread.start()

    def stop(self):
        with self.lock:
            if self.state != VideoPlayerState.STOPPED:
                logging.debug("STOPPING")
                self.state = VideoPlayerState.STOPPED
                self.playlist.clear()
                self.current_video = None
                if self.playback_thread.is_alive():
                    self.stop_event.set()
                    self.playback_thread.join(timeout=1.0)  # Provide a timeout so it doesn't wait indefinitely
                    if self.playback_thread.is_alive():
                        logging.warning("Playback thread failed to stop, may lead to unstable state.")

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
            self.fade_factor = 1.0
            self.fade_out()
            self.current_video_index = (self.current_video_index + 1) % len(self.playlist)
            self.load_video(self.current_video_index)
            self.fade_in()

    def prev_video(self):
        with self.lock:
            self.fade_factor = 1.0
            self.fade_out()
            self.current_video_index = (self.current_video_index - 1) % len(self.playlist)
            self.load_video(self.current_video_index)
            self.fade_in()

    def restart_video(self):
        with self.lock:
            self.fade_factor = 1.0
            self.fade_out()
            self.load_video(self.current_video_index)
            self.fade_in()

    def play_by_name(self, name: str) -> bool:
        with self.lock:
            for i, item in enumerate(self.playlist):
                if os.path.basename(item["filepath"]) == name:
                    self.fade_factor = 1.0
                    self.fade_out()
                    self.current_video_index = i
                    self.load_video(self.current_video_index)
                    self.fade_in()
                    return True
            return False

    def play_by_index(self, index: int) -> bool:
        with self.lock:
            if 0 <= index < len(self.playlist):
                self.fade_factor = 1.0
                self.fade_out()
                self.current_video_index = index
                self.load_video(self.current_video_index)
                self.fade_in()
                return True
            return False

    def load_video(self, index):
        playlist = self.playlist
        logging.debug("PLAYLIST %s", playlist)
        filepath = playlist[index]["filepath"]
        logging.debug("LOADING VIDEO %s", filepath)
        self.current_video = WWFile(filepath)
        self.fade_factor = 0.0
        self.fade_in()

    def playback_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                if self.state == VideoPlayerState.STOPPED:
                    break
                elif self.state == VideoPlayerState.PAUSED:
                    time.sleep(0.01)
                    continue
                elif self.state == VideoPlayerState.PLAYING:
                    if not self.playlist:
                        logging.debug("No playlist, loading")
                        self.load_playlist()

                    if not self.current_video and self.playlist:
                        logging.debug("No current video, loading")
                        self.load_video(self.current_video_index)

                    if self.current_video:
                        self.current_video.update()
                        frame = self.current_video.get_next_frame()
                        if frame is not None:
                            # apply fading to the frame here.
                            frame = self.apply_fade(frame)
                            if self.display_callback:
                                self.display_callback(frame)

                        else:
                            # Fade out
                            if self.fade_factor > 0.0:
                                self.fade_factor -= self.fade_speed
                            else:
                                self.current_video = None
                                if self.mode == VideoPlayerMode.REPEAT_ONE:
                                    self.restart_video()
                                elif self.mode == VideoPlayerMode.REPEAT:
                                    self.next_video()
                                elif self.mode == VideoPlayerMode.REPEAT_NONE:
                                    if self.current_video_index < len(self.playlist) - 1:
                                        self.next_video()
                                    else:
                                        self.stop()

            if self.stop_event.wait(1 / self.fps):
                # returns immediately if the event is set, else waits for the timeout
                logging.debug("Stop event set, breaking")
                break

    def apply_fade(self, frame):
        # Ease in and out of the fade
        fade = (1.0 - np.cos(self.fade_factor * np.pi)) / 2.0
        return np.round(frame * fade).astype(np.uint8)

    def fade_in(self):
        while self.fade_factor < 1.0:
            self.fade_factor += self.fade_speed
            time.sleep(0.01)
        self.fade_factor = 1.0

    def fade_out(self):
        while self.fade_factor > 0.0:
            self.fade_factor -= self.fade_speed
            time.sleep(0.01)
        self.fade_factor = 0.0

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
