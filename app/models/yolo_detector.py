from ultralytics import YOLO
from pathlib import Path


class YOLODetector:
    def __init__(self, model_path: Path, device: str):
        self.model = YOLO(str(model_path))
        self.model.to(device)

    def detect_persons(self, image_bgr):
        results = self.model(image_bgr)
        person_boxes = []
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                label = res.names[int(cls)]
                if label == "truck" or label == "car":
                    x1, y1, x2, y2 = map(int, box[:4])
                    person_boxes.append((x1, y1, x2, y2))
        return person_boxes
