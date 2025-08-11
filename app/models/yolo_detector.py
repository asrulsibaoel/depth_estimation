import numpy as np
from ultralytics import YOLO
from pathlib import Path


class YOLODetector:
    def __init__(self, model_path: Path, device: str):
        self.model = YOLO(str(model_path))
        self.model.to(device)

    def detect(self, image_bgr: np.ndarray, obj_class: str = "person"):
        results = self.model(image_bgr)
        target_boxes = []
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                label = res.names[int(cls)]
                if label == obj_class:
                    x1, y1, x2, y2 = map(int, box[:4])
                    target_boxes.append((x1, y1, x2, y2))
        return target_boxes
