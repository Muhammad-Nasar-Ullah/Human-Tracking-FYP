from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from tracker import process_video, reset_counters, set_line_position, get_counts
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

class LineSettings(BaseModel):
    line_y: int

@app.get("/stats")
def get_stats():
    return get_counts()

@app.post("/settings/line")
def update_line(settings: LineSettings):
    set_line_position(settings.line_y)
    return {"status": "ok", "line_y": settings.line_y}

def generate_frames(video_name: str):
    # Resolve absolute path to be safe
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "videos", video_name)

    for frame in process_video(video_path):
        # Encode with lower quality for speed (70%)
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

@app.get("/video")
def video_feed(video: str = Query("1.mp4")):
    reset_counters()  # important when switching videos
    return StreamingResponse(
        generate_frames(video),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
