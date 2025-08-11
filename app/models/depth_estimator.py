import torch
import depth_pro
from pathlib import Path


class DepthEstimator:
    def __init__(self, ckpt_path: Path, device: str):
        self.model, self.transform = depth_pro.create_model_and_transforms(
            ckpt_path=str(ckpt_path))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def infer_depth(self, image_path: str):
        image, _, f_px = depth_pro.load_rgb(image_path)
        inp = self.transform(image).to(self.device)
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)
        with torch.no_grad():
            prediction = self.model.infer(inp, f_px=f_px)
        return prediction["depth"].squeeze().cpu().numpy()
