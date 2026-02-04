"""
Human Tracking Module - Optimized Hybrid Detect-Then-Track
Connects Detection, Tracking, Counting, and Visualization modules.
"""

import cv2
import numpy as np
import time
from threading import Lock, Event

# Import new modules
from detection import get_detector
from tracking import ByteTracker
from counting import LineCounter
from visualization import draw_tracks, draw_line, draw_stats

# Configuration
DETECT_INTERVAL = 5 # Run detection every N frames
PROCESS_WIDTH = 640 # Resize width

class VideoTracker:
    """
    Manages tracking state for a single video stream.
    """

    def __init__(self, video_path: str, line_y_percent: float = 50.0):
        self.video_path = video_path
        
        # Modules
        self.detector = get_detector()
        self.tracker = ByteTracker(frame_rate=30)
        self.counter = LineCounter(line_y_percent)
        
        # State
        self.line_y_percent = line_y_percent
        self.process_width = PROCESS_WIDTH
        self.process_height = 360 # Will be updated based on aspect ratio
        
        self.original_width = 0
        self.original_height = 0
        self.fps = 30
        
        self.current_frame = 0 # Initialize
        
        self._lock = Lock()
        
        # Playback Control
        self.paused = False
        self.seek_frame = None # Frame index to seek to

    def set_line_position(self, y_percent: float):
        """Set line position as percentage of frame height."""
        with self._lock:
            self.line_y_percent = max(5.0, min(95.0, y_percent))
            # Update counter immediately if resolution is known
            if self.process_height > 0:
                self.counter.set_line_position(self.line_y_percent, self.process_height)

    def get_counts(self) -> dict:
        with self._lock:
            return self.counter.get_counts()

    def reset(self):
        with self._lock:
            self.tracker = ByteTracker(frame_rate=self.fps) # Re-init tracker
            self.counter.reset()
            self.paused = False
            self.seek_frame = None

    def pause(self):
        self.paused = True
        
    def resume(self):
        self.paused = False
        
    def seek(self, seconds: float):
        """Relative seek in seconds (+/-)."""
        if self.fps > 0:
            offset = int(seconds * self.fps)
            with self._lock:
               self.seek_frame = self.current_frame + offset

    def get_video_info(self) -> dict:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "process_width": self.process_width,
            "process_height": self.process_height,
            "fps": self.fps
        }

    def process_frames(self, skip_frames: int = 0):
        """
        Generator that yields processed frames.
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"[Tracker] Error: Could not open video file {self.video_path}")
            yield self._create_error_frame("Video Error")
            return

        # Get video properties
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Calculate processing dimensions
        aspect_ratio = self.original_height / self.original_width
        self.process_height = int(self.process_width * aspect_ratio)
        
        # Update counter line position with correct height
        self.counter.set_line_position(self.line_y_percent, self.process_height)

        print(f"[Tracker] Video: {self.original_width}x{self.original_height} @ {self.fps:.1f}fps")
        print(f"[Tracker] Processing at: {self.process_width}x{self.process_height}")

        self.current_frame = 0 # Track current frame index shared
        
        frame_idx = 0
        
        try:
            while True:
                self.current_frame = frame_idx
                
                # Handle Seek
                if self.seek_frame is not None:
                    with self._lock:
                        target_frame = max(0, self.seek_frame)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        frame_idx = target_frame
                        self.current_frame = frame_idx
                        self.seek_frame = None
                        # Optionally reset tracker state on large seek?
                        # self.tracker = ByteTracker(frame_rate=self.fps) 

                # Handle Pause
                while self.paused:
                    time.sleep(0.1)
                    # Yield last frame or keep alive? 
                    # MJPEG needs a frame to keep connection alive usually or just wait.
                    # Best to yield the same frame again if possible, or just wait.
                    # But if we wait too long, browser might timeout.
                    # Simple approach: Check state every 0.1s.
                    if self.seek_frame is not None:
                        break # Break pause loop to handle seek
                
                ret, frame = cap.read()

                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Reset tracker but keep counts
                    self.tracker = ByteTracker(frame_rate=self.fps) 
                    # Do NOT reset counter
                    print("[Tracker] Video looped.")
                    continue

                frame_idx += 1
                
                # Check for stop/skip? 
                # (For now we process every frame of the video stream for smoothness, 
                # but only run detection periodically)
                
                # Resize
                frame = cv2.resize(frame, (self.process_width, self.process_height))
                
                # Hybrid Logic
                tracks = []
                
                if frame_idx % DETECT_INTERVAL == 0:
                    # 1. Detect
                    detections = self.detector.detect(frame)
                    
                    # 2. Update Tracker
                    # ByteTracker expects specific format? 
                    # My detector returns [x1, y1, x2, y2, score, class]
                    # My ByteTracker implementation handles this (cols 0-4 are boxes+score)
                    tracks = self.tracker.update(detections, None, None)
                else:
                    # 1. Predict Only
                    tracks = self.tracker.predict()
                
                # 3. Update Counting
                with self._lock:
                    self.counter.update(tracks)
                    counts = self.counter.get_counts()
                    # Ensure line position is current (in case changed mid-stream)
                    self.counter.set_line_position(self.line_y_percent, self.process_height)
                
                # 4. Visualization
                # Only draw essential info
                draw_tracks(frame, tracks)
                draw_line(frame, self.counter.line_y, self.process_width)
                draw_stats(frame, counts["in"], counts["out"], fps=self.fps)

                yield frame

        finally:
            cap.release()

    def _create_error_frame(self, text):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, text, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame


# Global tracker registry
_trackers = {}
_trackers_lock = Lock()

def get_or_create_tracker(video_path: str, reset: bool = False) -> VideoTracker:
    with _trackers_lock:
        if video_path not in _trackers or reset:
            _trackers[video_path] = VideoTracker(video_path)
        return _trackers[video_path]

def get_current_tracker() -> VideoTracker | None:
    with _trackers_lock:
        if _trackers:
            return list(_trackers.values())[-1]
        return None

# Legacy/Bridge functions for app.py
def set_line_position(y_percent: float):
    tracker = get_current_tracker()
    if tracker:
        tracker.set_line_position(float(y_percent))

def get_counts() -> dict:
    tracker = get_current_tracker()
    if tracker:
        return tracker.get_counts()
    return {"in": 0, "out": 0}

def reset_counters():
    tracker = get_current_tracker()
    if tracker:
        tracker.reset()

def pause_video():
    tracker = get_current_tracker()
    if tracker:
        tracker.pause()

def resume_video():
    tracker = get_current_tracker()
    if tracker:
        tracker.resume()

def seek_video(seconds: float):
    tracker = get_current_tracker()
    if tracker:
        tracker.seek(seconds)

def process_video(video_path: str):
    tracker = get_or_create_tracker(video_path, reset=True)
    yield from tracker.process_frames()
