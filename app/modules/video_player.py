import os
import cv2
import glob
import time
import json
import threading
import numpy as np
from enum import Enum
from collections import deque
from typing import Callable, Optional, List, Dict

class VideoPlayerState(Enum):
    PLAYING = 1
    STOPPED = 2
    PAUSED = 3

class VideoPlayerMode(Enum):
    REPEAT = 1
    REPEAT_ONE = 2
    STOP = 3

class VideoPlayer:
    def __init__(self, ws_queue, video_dir: str, fps: int = 30, 
                 display_callback: Optional[Callable[[np.array], None]] = None):
        self.video_dir = video_dir
        self.ws_queue = ws_queue
        self.fps = fps
        self.state = VideoPlayerState.STOPPED
        self.mode = VideoPlayerMode.REPEAT
        self.current_video = None
        self.playlist: List[Dict[str, str]] = []
        self.playlist_path = os.path.join(video_dir, "playlist.json")
        self.lock = threading.Lock()
        self.current_video_index = 0
        self.display_callback = display_callback

        self.load_playlist()

    def play(self):
        with self.lock:
            if self.state != VideoPlayerState.PLAYING:
                self.state = VideoPlayerState.PLAYING
                if not hasattr(self, "playback_thread") or not self.playback_thread.is_alive():
                    self.playback_thread = threading.Thread(target=self.playback_loop)
                    self.playback_thread.start()

    def stop(self):
        with self.lock:
            if self.state != VideoPlayerState.STOPPED:
                self.state = VideoPlayerState.STOPPED
                self.playlist.clear()
                self.current_video = None
                if self.playback_thread and self.playback_thread.is_alive():
                    self.playback_thread.join()

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

    def load_video(self, index: int):
        with self.lock:
            self.current_video_index = index
            self.current_video = cv2.VideoCapture(self.playlist[self.current_video_index]['video_path'])

    def load_playlist(self):
        if not os.path.exists(self.playlist_path):
            self.create_playlist_file()
        
        with open(self.playlist_path, "r") as f:
            playlist = json.load(f)
        self.mode = VideoPlayerMode[playlist["mode"].upper()]
        self.playlist = [
            {"video_path": os.path.join(self.video_dir, video_path), "playback_mode": self.mode}
            for video_path in playlist["playlist"]
        ]
            
    def create_playlist_file(self):
        video_files = glob.glob(os.path.join(self.video_dir, "*"))
        video_files = [file for file in video_files if file.endswith((".mp4", ".mov", ".avi"))]
        playlist = {
            "mode": self.mode.name,
            "playlist": [os.path.basename(video_file) for video_file in video_files]
        }
        with open(self.playlist_path, "w") as f:
            json.dump(playlist, f, indent=4)

    def playback_loop(self):
        while True:
            if self.state == VideoPlayerState.STOPPED:
                break

            elif self.state == VideoPlayerState.PAUSED:
                time.sleep(0.01)
                continue

            elif self.state == VideoPlayerState.PLAYING:
                if not self.current_video and self.playlist:
                    self.load_video(self.current_video_index)

                if self.current_video:
                    ret, frame = self.current_video.read()
                    if ret:
                        self.display_callback(frame)
                    else:
                        self.current_video.release()
                        self.current_video = None
                        if self.mode == VideoPlayerMode.REPEAT_ONE:
                            self.restart_video()
                        elif self.mode == VideoPlayerMode.REPEAT:
                            self.next_video()
                        elif self.mode == VideoPlayerMode.STOP:
                            self.stop()
            time.sleep(1 / self.fps)

