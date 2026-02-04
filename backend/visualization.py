"""
Visualization Module
Handles drawing of tracks, counts, and UI elements.
Optimized to reduce processing time.
"""

import cv2
import numpy as np

def draw_tracks(frame, tracks, history=None):
    """
    Draw bounding boxes and IDs for tracks.
    
    Args:
        frame: Image to draw on
        tracks: List of Track objects (expected to have track_id, tlbr)
        history: Optional dict of track histories {track_id: list of points}
    """
    for track in tracks:
        if not track.is_activated:
            continue
            
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.tlbr)
        
        # Color based on ID
        color = _get_color(track_id)
        
        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        label = f"ID:{track_id}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw Center
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

def draw_line(frame, line_y, width):
    """Draw the counting line."""
    # Main line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
    
    # Label
    cv2.putText(frame, "COUNTING LINE", (10, line_y - 8), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def draw_stats(frame, in_count, out_count, fps=None):
    """Draw counters and usage stats."""
    # Background for stats
    # cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), 1)
    
    # IN Count
    cv2.putText(frame, f"IN: {in_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # OUT Count
    cv2.putText(frame, f"OUT: {out_count}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
               
    # FPS if provided
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def _get_color(idx):
    """Get a consistent color for an ID."""
    idx = idx * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
