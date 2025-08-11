from app.config import DEPTH_MODEL_TYPE
from app.models.midas_estimator import MiDaSEstimator
from app.models.depth_pro_estimator import DepthProEstimator


def get_depth_estimator_instance():
    if DEPTH_MODEL_TYPE == "midas":
        return MiDaSEstimator()
    elif DEPTH_MODEL_TYPE == "depth_pro":
        return DepthProEstimator()
    else:
        raise ValueError(f"Unsupported depth model: {DEPTH_MODEL_TYPE}")
