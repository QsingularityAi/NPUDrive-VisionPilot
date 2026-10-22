"""AutoDrive Model: 2-Frame Longitudinal & Curvature Perception."""
import cv2
import numpy as np
from .base_model import BaseTIDLModel
from ..config.config import VisionPilotConfig

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-float(x)))

class AutoDriveModel(BaseTIDLModel):
    def __init__(self, onnx_path: str, artifacts_dir: str, prefer_tidl: bool = True):
        super().__init__("autodrive", onnx_path, artifacts_dir, prefer_tidl=prefer_tidl)
        self.prev_frame_chw = None
        self.frame_buffer_ready = False

    def preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        """Center-crops to 50 deg HFOV, resizes to NET_W x NET_H, converts BGR->RGB, applies ImageNet normalization."""
        h, w = bgr_img.shape[:2]
        crop_w = int(round(w * 50.0 / 120.0))
        crop_h = crop_w // 2
        crop_w = max(1, min(crop_w, w))
        crop_h = max(1, min(crop_h, h))
        x0 = max(0, (w - crop_w) // 2)
        y0 = max(0, (h - crop_h) // 2)
        crop = bgr_img[y0:y0 + crop_h, x0:x0 + crop_w]
        resized = cv2.resize(crop, (VisionPilotConfig.NET_W, VisionPilotConfig.NET_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        # Transpose to CHW and normalize
        x = (x.transpose(2, 0, 1)[None] - VisionPilotConfig.IMAGENET_MEAN) / VisionPilotConfig.IMAGENET_STD
        return np.ascontiguousarray(x, dtype=np.float32)

    def infer(self, curr_bgr: np.ndarray) -> dict:
        curr_chw = self.preprocess(curr_bgr)
        
        if self.prev_frame_chw is None:
            self.prev_frame_chw = curr_chw.copy()
            self.frame_buffer_ready = False
            return {
                "valid": False,
                "distance_m": VisionPilotConfig.AUTODRIVE_D_MAX_M,
                "distance_norm": 0.0,
                "curvature": 0.0,
                "flag_prob": 0.0,
                "latency_ms": 0.0
            }

        feed = {
            self.input_names[0]: self.prev_frame_chw,
            self.input_names[1]: curr_chw
        }
        
        outs = self.run(feed)
        self.prev_frame_chw = curr_chw.copy()
        self.frame_buffer_ready = True

        # Outputs: distance (1, 1), curvature (1, 1), flag_logit (1, 1)
        dist_norm = float(outs[0].squeeze())
        curv_raw  = float(outs[1].squeeze())
        flag_logit = float(outs[2].squeeze())

        dist_m = VisionPilotConfig.AUTODRIVE_D_MAX_M * (1.0 - dist_norm)
        dist_m = float(np.clip(dist_m, 0.0, VisionPilotConfig.AUTODRIVE_D_MAX_M))
        flag_prob = sigmoid(flag_logit)

        return {
            "valid": True,
            "distance_m": dist_m,
            "distance_norm": dist_norm,
            "curvature": curv_raw * VisionPilotConfig.AUTODRIVE_CURV_SCALE,
            "flag_prob": flag_prob,
            "has_obstacle": flag_prob >= VisionPilotConfig.AUTODRIVE_FLAG_THRES,
            "latency_ms": self.last_latency_ms
        }

    def reset(self):
        self.prev_frame_chw = None
        self.frame_buffer_ready = False
