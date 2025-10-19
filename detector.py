from ultralytics import YOLO
import os


class ObjectDetection:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        return YOLO('yolo11n.pt')


    def detect(self, frame):
        results = self.model.predict(source=frame, imgsz=1280, conf=0.5)

        # Extract detections
        detections = results[0]
        boxes = detections.boxes.xyxy.cpu().numpy()
        class_ids = detections.boxes.cls.cpu().numpy().astype(int)
        scores = detections.boxes.conf.cpu().numpy()

        return class_ids, scores, boxes