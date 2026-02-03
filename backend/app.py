"""
FastAPI Backend for Human Tracking System
Provides video streaming with real-time detection and counting.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2
import os
import time

from tracker import (
    process_video,
    reset_counters,
    set_line_position,
    get_counts,
    get_current_tracker,
    get_or_create_tracker
)

app = FastAPI(
    title="Human Tracking API",
    description="Real-time human detection and counting system",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
JPEG_QUALITY = 85  # Higher quality for better visualization


class LineSettings(BaseModel):
    """Settings for the counting line position."""
    line_percent: float  # Percentage of frame height (0-100)




# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "online", "service": "Human Tracking API", "version": "2.0.0"}


@app.get("/stats")
def get_stats():
    """Get current IN/OUT counts."""
    return get_counts()


@app.get("/videos")
def list_videos():
    """List available video files from the videos folder."""
    if not os.path.exists(VIDEOS_DIR):
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        return {"videos": [], "message": "Videos folder created. Add video files to backend/videos/"}

    videos = []
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')

    for filename in os.listdir(VIDEOS_DIR):
        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(VIDEOS_DIR, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            videos.append({
                "name": filename,
                "size_mb": round(size_mb, 2)
            })

    # Sort videos alphabetically by name
    videos.sort(key=lambda x: x["name"].lower())

    return {"videos": videos, "count": len(videos)}


@app.get("/video/info")
def get_video_info():
    """Get current video metadata."""
    tracker = get_current_tracker()
    if tracker:
        return tracker.get_video_info()
    return {
        "original_width": 0,
        "original_height": 0,
        "process_width": 640,
        "process_height": 360,
        "fps": 30
    }


@app.post("/settings/line")
def update_line(settings: LineSettings):
    """Update counting line position as percentage of frame height (0-100)."""
    set_line_position(settings.line_percent)
    return {"status": "ok", "line_percent": settings.line_percent}




@app.post("/reset")
def reset_tracking():
    """Reset tracking counters and history."""
    reset_counters()
    return {"status": "ok", "message": "Counters reset"}


def generate_frames(video_name: str):
    """
    Generator for MJPEG streaming.
    Yields frames as multipart data with proper boundaries.
    """
    video_path = os.path.join(VIDEOS_DIR, video_name)

    if not os.path.exists(video_path):
        # Yield error frame
        import numpy as np
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Video not found: {video_name}", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, buffer = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )
        return

    frame_count = 0
    start_time = time.time()

    for frame in process_video(video_path):
        frame_count += 1

        # Encode frame as JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)

        if not success:
            continue

        # Yield as multipart MIME
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        # Log FPS periodically
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[Stream] Streaming at {fps:.1f} FPS")


@app.get("/video")
def video_feed(video: str = Query("1.mp4")):
    """
    Stream video with real-time human tracking overlay.
    Uses MJPEG format for browser compatibility.
    """
    # Validate video filename (security)
    if ".." in video or "/" in video or "\\" in video:
        raise HTTPException(status_code=400, detail="Invalid video name")

    # Reset counters when switching videos
    reset_counters()

    return StreamingResponse(
        generate_frames(video),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive"
        }
    )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Human Tracking API Server...")
    print(f"Videos directory: {VIDEOS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
