from functools import lru_cache
from app.config import settings
from app.models.yolo_detector import YOLODetector
from app.models.depth_estimator import DepthEstimator
from app.models.midas_estimator import MiDaSEstimator
from app.services.distance_service import DistanceService


@lru_cache
def get_yolo_detector() -> YOLODetector:
    return YOLODetector(model_path=settings.yolo_model_path, device=settings.device)


@lru_cache
def get_depth_estimator():
    if settings.depth_model_type == "midas":
        return MiDaSEstimator(model_type=settings.depth_model_name, device=settings.device)
    elif settings.depth_model_type == "depth_pro":
        return DepthEstimator(ckpt_path=settings.depth_pro_ckpt, device=settings.device)
    else:
        raise ValueError(
            f"Unsupported DEPTH_MODEL_TYPE: {settings.depth_model_type}")


@lru_cache
def get_distance_service() -> DistanceService:
    detector = YOLODetector(
        model_path=settings.yolo_model_path, device=settings.device)
    depth_model = get_depth_estimator()
    return DistanceService(detector=detector, depth_model=depth_model)
