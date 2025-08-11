from pathlib import Path
from typing import Literal
import torch
from pydantic_settings import BaseSettings  # <- from pydantic, no extra install


class Settings(BaseSettings):
    base_dir: Path = Path(__file__).resolve().parent.parent
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model_path: Path = base_dir / "checkpoints" / "yolov8n.pt"
    depth_model_type: Literal["midas", "depth_pro"] = "midas"
    depth_model_name: str = "DPT_Hybrid"
    depth_pro_ckpt: Path = base_dir / "checkpoints" / "depth_pro.pt"

    avg_human_height_m: float = 1.75  # Average human height in meters

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
