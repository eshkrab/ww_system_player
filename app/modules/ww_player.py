import os
import cv2
import glob
import time
import array
import json
import threading
import logging
import socket

import cProfile
import pstats
import io


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

    def __init__(self, ws_queue, video_dir: str, fps: int = 60, display_callback: Optional[Callable[[np.array], None]] = None):
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
        self.last_fps_print_time = time.time()  # Initialize the attribute

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
                    #  self.playback_thread = threading.Thread(target=self.send_dummy_data)
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
            #  self.fade_factor = 1.0
            #  self.fade_out()
            logging.debug("NEXT VIDEO")
            logging.debug("playlist len %d", len(self.playlist))
            self.current_video_index = (self.current_video_index + 1) % len(self.playlist)
            self.load_video(self.current_video_index)
            #  self.fade_in()

    def prev_video(self):
        with self.lock:
            #  self.fade_factor = 1.0
            #  self.fade_out()
            self.current_video_index = (self.current_video_index - 1) % len(self.playlist)
            self.load_video(self.current_video_index)
            #  self.fade_in()

    def restart_video(self):
        with self.lock:
            #  self.fade_factor = 1.0
            #  self.fade_out()
            self.load_video(self.current_video_index)
            #  self.fade_in()

    def play_by_name(self, name: str) -> bool:
        with self.lock:
            for i, item in enumerate(self.playlist):
                if os.path.basename(item["filepath"]) == name:
                    #  self.fade_factor = 1.0
                    #  self.fade_out()
                    self.current_video_index = i
                    self.load_video(self.current_video_index)
                    #  self.fade_in()
                    return True
            return False

    def play_by_index(self, index: int) -> bool:
        with self.lock:
            if 0 <= index < len(self.playlist):
                #  self.fade_factor = 1.0
                #  self.fade_out()
                self.current_video_index = index
                self.load_video(self.current_video_index)
                #  self.fade_in()
                return True
            return False

    def load_video(self, index):
        playlist = self.playlist['playlist']
        logging.debug("PLAYLIST %s", playlist)
        logging.debug("INDEX %s", index)
        logging.debug("PLAYLIST ITEM %s", playlist[index])
        filepath = playlist[index]["filepath"]
        logging.debug("LOADING VIDEO %s", filepath)
        self.current_video = WWFile(filepath)
        #  self.fade_factor = 0.0
        #  self.fade_in()

    def print_stats(self, pr):
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


    def playback_loop(self):
        fps_history = deque(maxlen=self.fps * 60)  # Keep track of fps for the last minute
        start_time = time.monotonic()
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
                            if self.display_callback:
                                callback_start_time = time.monotonic()
                                self.display_callback(frame)
                                callback_end_time = time.monotonic()
                                callback_time = callback_end_time - callback_start_time

                        else:
                            logging.debug("Frame is None")
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

                        #  time.sleep(1/self.fps)

            #  logging.debug(f"Frame took {end_time - start_time:.3f} seconds")
            #  logging.debug(f"Callback took {callback_time:.3f} seconds")
            # Measure fps
            end_time = time.monotonic()
            frame_time = end_time - start_time  # Time taken to process and display the frame

            # Sleep for the remaining time in the frame, if any
            remaining_time = (1/self.fps) - frame_time
            if remaining_time > 0:
                time.sleep(remaining_time)

            fps = 1 / (time.monotonic() - start_time)  # Recompute fps after the sleep
            fps_history.append(fps)

            # Print fps every minute
            if time.time() - self.last_fps_print_time >= 30:
                avg_fps = sum(fps_history) / len(fps_history)
                logging.debug(f"Average fps for the last minute: {avg_fps:.2f}")
                fps_history.clear()
                self.last_fps_print_time = time.time()
                

            # Returns immediately if the clear event is set, else waits for the timeout
            if self.stop_event.wait(1 / self.fps):
                logging.debug("Stop event set, breaking")
                break

            start_time = time.monotonic()

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

    def send_dummy_data(self):
        start_time = time.time()
        while True:
            # Calculate rainbow fade from time
            elapsed_time = time.time() - start_time
            rainbow = np.array([[(np.sin(0.3 * i + elapsed_time) + 1) * 127 for i in range(300)]], dtype=np.uint8)
            # Send dummy data to display_callback
            if self.display_callback:
                if isinstance(rainbow, np.ndarray):
                    # Convert numpy array to a list of integers
                    scaled_data = [round(byte * self.fade_factor) for byte in rainbow[0]]
                    self.display_callback(scaled_data)
                else:
                    self.display_callback(rainbow)
            # Wait 1/30 seconds (assuming fps = 30)
            time.sleep(1 / self.fps)
