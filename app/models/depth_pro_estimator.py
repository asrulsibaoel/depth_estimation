import torch
import depth_pro
from app.config import DEVICE, DEPTH_PRO_CKPT


class DepthProEstimator:
    def __init__(self):
        self.model, self.transform = depth_pro.create_model_and_transforms(
            weights=DEPTH_PRO_CKPT)
        self.model.to(DEVICE).eval()

    def infer_depth(self, image_path: str):
        image, _, f_px = depth_pro.load_rgb(image_path)
        inp = self.transform(image)
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)
        inp = inp.to(DEVICE)

        with torch.no_grad():
            prediction = self.model.infer(inp, f_px=f_px)

        return prediction["depth"].squeeze().cpu().numpy()
