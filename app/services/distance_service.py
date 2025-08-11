from typing import List, Dict

import cv2
import numpy as np
import math


class DistanceService:
    def __init__(self, detector, depth_model):
        self.detector = detector
        self.depth_model = depth_model

    def _compute_focal_length(self, img_width: int, camera_fov_deg: float) -> float:
        fov_rad = math.radians(camera_fov_deg)
        return img_width / (2 * math.tan(fov_rad / 2))

    def process_image(
        self,
        image_path: str,
        camera_pov: float,
        average_height: float = 1.7,
        obj_class: str = "person"
    ) -> List[Dict]:
        image = cv2.imread(image_path)
        detections = self.detector.detect(image, obj_class=obj_class)
        depth_map = self.depth_model.infer_depth(image_path)
        img_width = image.shape[1]

        focal_length_px = self._compute_focal_length(img_width, camera_pov)

        results = []
        for det in detections:
            # bbox = det["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = det
            pixel_height = max(1, y2 - y1)

            # Base distance using human height baseline
            distance_m = (average_height *
                          focal_length_px) / pixel_height

            # Optional refinement from depth map
            median_depth = float(np.median(depth_map[y1:y2, x1:x2]))
            depth_scale = np.mean(depth_map) / median_depth
            distance_m *= depth_scale

            results.append({
                "bbox": det,
                "distance_m": distance_m
            })

        return results
