"""
Detection Module
Wraps YOLOv8 for person detection.
"""

import os
import torch
from ultralytics import YOLO
import numpy as np

# Global configuration
MODEL_NAME = "yolov8n.pt"  # Use nano model for speed
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

class Detector:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Detector, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """Initialize the YOLO model."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, MODEL_NAME)
        
        print(f"[Detector] Loading model from {model_path}...")
        self._model = YOLO(model_path)
        
        # Optimize for inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Using device: {self.device}")

    def detect(self, frame, classes=[0]):
        """
        Run detection on a frame.
        
        Args:
            frame: Numpy array (image)
            classes: List of class IDs to detect (default [0] for person)
            
        Returns:
            np.ndarray: Detections in format [[x1, y1, x2, y2, score, class_id], ...]
        """
        # Run inference
        results = self._model(
            frame, 
            classes=classes,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=self.device
        )
        
        result = results[0]
        detections = []
        
        if result.boxes:
            # Get boxes, scores, lists
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            
            # Stack into single array
            for box, score, cls_id in zip(boxes, scores, cls_ids):
                detections.append([*box, score, cls_id])
                
        return np.array(detections) if detections else np.empty((0, 6))

# Helper function to get singleton
def get_detector():
    return Detector()
