"""
Human Tracking Module - Optimized for Real-time Video Processing
Uses YOLOv8 with ByteTrack for person detection and tracking.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from threading import Lock
import os

# Load model once at module level (lazy loading would add latency on first request)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
model = YOLO(MODEL_PATH)

# Check if CUDA is available for GPU acceleration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Tracker] Using device: {DEVICE}")


class VideoTracker:
    """
    Manages tracking state for a single video stream.
    Each video gets its own tracker instance to avoid state conflicts.
    """

    def __init__(self, video_path: str, line_y_percent: float = 50.0, process_width: int = 640):
        """
        Initialize tracker for a video.

        Args:
            video_path: Path to video file
            line_y_percent: Line position as percentage of frame height (0-100)
            process_width: Width to resize frames for processing (smaller = faster)
        """
        self.video_path = video_path
        self.line_y_percent = line_y_percent
        self.process_width = process_width

        # Tracking state
        self.track_history = {}  # track_id -> last y position
        self.in_count = 0
        self.out_count = 0

        # Video info (set when processing starts)
        self.original_width = 0
        self.original_height = 0
        self.process_height = 0
        self.fps = 30

        # Thread safety
        self._lock = Lock()

    def set_line_position(self, y_percent: float):
        """Set line position as percentage of frame height."""
        with self._lock:
            self.line_y_percent = max(5.0, min(95.0, y_percent))

    def get_counts(self) -> dict:
        """Get current IN/OUT counts."""
        with self._lock:
            return {"in": self.in_count, "out": self.out_count}

    def reset(self):
        """Reset tracking state."""
        with self._lock:
            self.track_history.clear()
            self.in_count = 0
            self.out_count = 0

    def get_video_info(self) -> dict:
        """Get video metadata."""
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "process_width": self.process_width,
            "process_height": self.process_height,
            "fps": self.fps
        }

    def process_frames(self, skip_frames: int = 0):
        """
        Generator that yields processed frames with tracking.

        Args:
            skip_frames: Number of frames to skip between processed frames (0 = process all)
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"[Tracker] Error: Could not open video file {self.video_path}")
            # Yield a black frame with error message
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Video Error", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            yield error_frame
            return

        # Get video properties
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Calculate processing dimensions (maintain aspect ratio)
        aspect_ratio = self.original_height / self.original_width
        self.process_height = int(self.process_width * aspect_ratio)

        print(f"[Tracker] Video: {self.original_width}x{self.original_height} @ {self.fps:.1f}fps")
        print(f"[Tracker] Processing at: {self.process_width}x{self.process_height}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    # Loop video - only clear track history, keep counts!
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self._lock:
                        self.track_history.clear()  # Clear history but KEEP counts
                    print(f"[Tracker] Video looped. Counts preserved - IN: {self.in_count}, OUT: {self.out_count}")
                    ret, frame = cap.read()
                    if not ret:
                        break

                frame_count += 1

                # Skip frames if configured (for performance)
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue

                # Process frame
                processed_frame = self._process_single_frame(frame)
                yield processed_frame

        finally:
            cap.release()

    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with detection and tracking."""

        # Resize for processing
        frame = cv2.resize(frame, (self.process_width, self.process_height))

        # Calculate line Y position based on percentage
        with self._lock:
            line_y = int(self.process_height * (self.line_y_percent / 100.0))
            line_y = max(10, min(self.process_height - 10, line_y))

        # Run YOLO detection and tracking
        # classes=[0] filters for person class only
        results = model.track(
            frame,
            persist=True,
            classes=[0],  # Person class only
            verbose=False,
            device=DEVICE
        )

        # Process detections
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            with self._lock:
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cy = int((y1 + y2) / 2)  # Center Y of bounding box
                    cx = int((x1 + x2) / 2)  # Center X of bounding box

                    # Check line crossing
                    if track_id in self.track_history:
                        prev_y = self.track_history[track_id]

                        # Crossed from top to bottom (IN) - entering the area below the line
                        # prev_y was above line (< line_y) and now cy is below line (> line_y)
                        if prev_y < line_y and cy > line_y:
                            self.in_count += 1
                            print(f"[Tracker] IN detected! ID:{track_id} crossed line. Total IN: {self.in_count}")
                        # Crossed from bottom to top (OUT) - exiting to above the line
                        # prev_y was below line (> line_y) and now cy is above line (< line_y)
                        elif prev_y > line_y and cy < line_y:
                            self.out_count += 1
                            print(f"[Tracker] OUT detected! ID:{track_id} crossed line. Total OUT: {self.out_count}")

                    # Update history
                    self.track_history[track_id] = cy

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw ID label
                    label = f"ID:{track_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 8),
                                 (x1 + label_size[0] + 4, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Draw center point
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

                    # Draw tracking line from center to the counting line (visual feedback)
                    line_color = (0, 255, 255)  # Yellow
                    if cy < line_y:
                        # Person is above the line
                        cv2.line(frame, (cx, cy), (cx, line_y), (100, 100, 255), 1)  # Red-ish line down
                    else:
                        # Person is below the line
                        cv2.line(frame, (cx, cy), (cx, line_y), (100, 255, 100), 1)  # Green-ish line up

        # Draw counting line
        cv2.line(frame, (0, line_y), (self.process_width, line_y), (0, 255, 255), 2)

        # Draw line label
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw counts on frame
        with self._lock:
            in_count = self.in_count
            out_count = self.out_count

        cv2.putText(frame, f"IN: {in_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {out_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        return frame


# Global tracker registry for managing multiple video streams
_trackers = {}
_trackers_lock = Lock()


def get_or_create_tracker(video_path: str, reset: bool = False) -> VideoTracker:
    """Get existing tracker or create new one for a video."""
    with _trackers_lock:
        if video_path not in _trackers or reset:
            _trackers[video_path] = VideoTracker(video_path)
        return _trackers[video_path]


def get_current_tracker() -> VideoTracker | None:
    """Get the most recently used tracker."""
    with _trackers_lock:
        if _trackers:
            return list(_trackers.values())[-1]
        return None


# Legacy API compatibility functions
_current_video_path = None


def set_line_position(y_percent: float):
    """Set line position as percentage (0-100)."""
    tracker = get_current_tracker()
    if tracker:
        tracker.set_line_position(float(y_percent))


def get_counts() -> dict:
    """Legacy API: Get current counts."""
    tracker = get_current_tracker()
    if tracker:
        return tracker.get_counts()
    return {"in": 0, "out": 0}


def reset_counters():
    """Legacy API: Reset counters."""
    tracker = get_current_tracker()
    if tracker:
        tracker.reset()


def process_video(video_path: str):
    """Legacy API: Process video and yield frames."""
    global _current_video_path
    _current_video_path = video_path
    tracker = get_or_create_tracker(video_path, reset=True)
    yield from tracker.process_frames()
