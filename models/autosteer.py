"""AutoSteer Model: Ego Path Trajectory Perception."""
import cv2
import numpy as np
from .base_model import BaseTIDLModel
from ..config.config import VisionPilotConfig
from ..calibration.homography import top_crop_2_1

class AutoSteerModel(BaseTIDLModel):
    def __init__(self, onnx_path: str, artifacts_dir: str, prefer_tidl: bool = True):
        super().__init__("autosteer", onnx_path, artifacts_dir, prefer_tidl=prefer_tidl)
        self.waypoint_y = np.linspace(0, VisionPilotConfig.NET_H - 1, VisionPilotConfig.AUTOSTEER_NUM_WAYPOINTS, dtype=int)

    def preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        """Top-crops frame to 2:1 aspect ratio, resizes to 1024x512, normalizes RGB to [0, 1]."""
        h, w = bgr_img.shape[:2]
        crop_top = top_crop_2_1(h, w)
        cropped = bgr_img[crop_top:, :]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (VisionPilotConfig.NET_W, VisionPilotConfig.NET_H), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

    def infer(self, curr_bgr: np.ndarray) -> dict:
        inp = self.preprocess(curr_bgr)
        outs = self.run({self.input_names[0]: inp})
        
        # outs[0]: lane_value (1, 1, 64, 1), outs[1]: height (1, 1, 64, 1)
        lane_val = outs[0].squeeze()   # (64,) normalized x [0, 1]
        height_val = outs[1].squeeze() # (64,) confidence score

        valid_mask = height_val >= VisionPilotConfig.AUTOSTEER_CONF_THRES
        valid_indices = np.where(valid_mask)[0]

        # Extract model-space waypoints (u_px, v_px)
        model_pts = []
        norm_pts = []
        for idx in valid_indices:
            u_norm = float(lane_val[idx])
            v_norm = float(self.waypoint_y[idx]) / float(VisionPilotConfig.NET_H)
            u_px = u_norm * VisionPilotConfig.NET_W
            v_px = float(self.waypoint_y[idx])
            model_pts.append((u_px, v_px))
            norm_pts.append((u_norm, v_norm))

        return {
            "valid": len(model_pts) >= 4,
            "lane_value": lane_val,
            "height": height_val,
            "model_pts": model_pts,       # (u, v) in 1024x512 space
            "norm_pts": norm_pts,         # (x_norm, y_norm) [0, 1]
            "num_valid_waypoints": len(model_pts),
            "total_waypoints": VisionPilotConfig.AUTOSTEER_NUM_WAYPOINTS,
            "latency_ms": self.last_latency_ms
        }
