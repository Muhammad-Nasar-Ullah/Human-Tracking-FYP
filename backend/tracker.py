track_history = {}
IN_COUNT = 0
OUT_COUNT = 0
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

LINE_Y = 300
track_history = {}
IN_COUNT = 0
OUT_COUNT = 0

def set_line_position(y: int):
    global LINE_Y
    LINE_Y = y


def get_counts():
    return {"in": IN_COUNT, "out": OUT_COUNT}

def process_video(video_path):
    global IN_COUNT, OUT_COUNT
    print(f"Attempting to open video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended, looping...")
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Read again to ensure we have a frame
            ret, frame = cap.read()
            if not ret:
                 print("Error: Could not read frame after looping.")
                 break

        # Resize for better FPS (Standard YOLO size)
        height, width = frame.shape[:2]
        new_width = 640
        new_height = int(height * (new_width / width))
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Recalculate line Y relative to new size
        # Ensure LINE_Y is scaled or handled correctly. 
        # Since LINE_Y is a pixel value adjusted by frontend on a potentially different size,
        # we need to be careful. The frontend slider is 50-600.
        # If the user sets 300, it expects 300px from top. 
        # With 640 width, height might be ~360. 300 is near bottom.
        current_line_y = min(max(LINE_Y, 10), new_height - 10)

        results = model.track(frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy,
                                     results[0].boxes.id):

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                track_id = int(track_id)

                if track_id in track_history:
                    prev_y = track_history[track_id]

                    # Use current_line_y for logic
                    if prev_y < current_line_y and cy > current_line_y:
                        IN_COUNT += 1
                    elif prev_y > current_line_y and cy < current_line_y:
                        OUT_COUNT += 1

                track_history[track_id] = cy

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                # cv2.putText(frame, f"ID {track_id}", (x1,y1-10),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.line(frame, (0,current_line_y), (frame.shape[1],current_line_y), (0,255,255), 2)
        
        yield frame

    cap.release()
def reset_counters():
    global track_history, IN_COUNT, OUT_COUNT
    track_history = {}
    IN_COUNT = 0
    OUT_COUNT = 0
