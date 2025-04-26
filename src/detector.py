from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path="yolov8m.pt"):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect(self, frame, conf_thresh=0.7):
        results = self.model.predict(frame, verbose=False, conf=conf_thresh, device=self.device)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        return boxes, classes, confs