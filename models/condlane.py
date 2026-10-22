"""CondLaneNet-small Lane Detection Model (TIDL NPU + Range MLP)."""
import os
import cv2
import numpy as np
from .base_model import BaseTIDLModel
from ..config.config import VisionPilotConfig

_COL = np.arange(VisionPilotConfig.CONDLANE_NET_W // 8, dtype=np.float32)  # 100
_ROW = np.arange(VisionPilotConfig.CONDLANE_NET_H // 8, dtype=np.float32)  # 40
COORDS = np.stack([
    np.broadcast_to(_COL / 100.0, (40, 100)),
    np.broadcast_to(_ROW[:, None] / 100.0, (40, 100))
])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def seeds_from_heatmap(hm_logits, thr=0.50):
    hm = np.clip(sigmoid(hm_logits[0, 0]), 1e-4, 1.0 - 1e-4)
    pad = np.pad(hm, 1, constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(pad, (3, 3)).max(axis=(2, 3))
    keep = np.where(win == hm, hm, 0.0)
    ys, xs = np.where(keep > thr)
    return [{"coord": (int(x), int(y)), "score": float(hm[y, x])} for y, x in zip(ys, xs)]

def nms_seeds_tiny(seeds, thr=4):
    groups = []
    for idx, s in enumerate(seeds):
        for g in groups:
            if any(np.hypot(s["coord"][0] - seeds[j]["coord"][0],
                            s["coord"][1] - seeds[j]["coord"][1]) <= thr for j in g):
                g.append(idx)
                break
        else:
            groups.append([idx])
    return [seeds[g[-1]] for g in groups]

def dynamic_conv(feat, w, b):
    """[64, H, W] features + [66] kernel (2 coord + 64 features) + bias -> [H, W]."""
    return np.tensordot(w[:2], COORDS, 1) + np.tensordot(w[2:], feat, 1) + b

def lane_range(mask, mlp):
    """Range MLP: mask [H, W] -> valid flag per row [H]."""
    h = np.maximum(mlp["w0"] @ mask.T + mlp["b0"][:, None], 0.0)
    logits = mlp["w1"] @ h + mlp["b1"][:, None]
    return logits.argmax(0)

def row_span(valid):
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return None
    return max(0, idx[0] - 1), min(39, idx[-1] + 1)

class CondLaneModel(BaseTIDLModel):
    OUTPUT_NAMES = ("hm", "mask_feat", "mask_w", "mask_b", "reg_w", "reg_b")
    SLOT_NAMES = {0: "ego-left", 1: "ego-right", 2: "left-outer", 3: "right-outer"}

    def __init__(self, onnx_path: str, artifacts_dir: str, mlp_path: str, prefer_tidl: bool = True):
        super().__init__("condlane", onnx_path, artifacts_dir, prefer_tidl=prefer_tidl)
        
        # Load MLP weights
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(f"Range MLP weights not found: {mlp_path}")
        self.mlp = dict(np.load(mlp_path))
        self.roi = VisionPilotConfig.CONDLANE_ROI

    def preprocess(self, bgr_img: np.ndarray, roi: tuple = None) -> tuple:
        """Crops ROI from camera frame, resizes to 800x320 uint8, returns NCHW [1, 3, 320, 800]."""
        if roi is None: roi = self.roi
        h, w = bgr_img.shape[:2]
        x0 = int(round(roi[0] * w))
        y0 = int(round(roi[1] * h))
        x1 = int(round(roi[2] * w))
        y1 = int(round(roi[3] * h))
        
        roi_crop = bgr_img[y0:y1, x0:x1]
        net = cv2.resize(roi_crop, (VisionPilotConfig.CONDLANE_NET_W, VisionPilotConfig.CONDLANE_NET_H),
                         interpolation=cv2.INTER_AREA)
        u8 = np.ascontiguousarray(net.transpose(2, 0, 1)[None], dtype=np.uint8)
        return u8, (x0, y0, x1, y1)

    def infer(self, curr_bgr: np.ndarray,
              hm_thr: float = None,
              nms_thr: int = None,
              roi: tuple = None) -> dict:
        
        if hm_thr is None: hm_thr = VisionPilotConfig.CONDLANE_HM_THR
        if nms_thr is None: nms_thr = VisionPilotConfig.CONDLANE_NMS_THR
        if roi is None: roi = self.roi

        raw_h, raw_w = curr_bgr.shape[:2]
        u8_inp, (rx0, ry0, rx1, ry1) = self.preprocess(curr_bgr, roi)

        outs_list = self.run({self.input_names[0]: u8_inp})
        out_dict = dict(zip(self.OUTPUT_NAMES, outs_list))

        # ─── Post-processing: Decode seeds & dynamic lanes ──────────────────
        raw_seeds = seeds_from_heatmap(out_dict["hm"], thr=hm_thr)
        grouped_seeds = nms_seeds_tiny(raw_seeds, thr=nms_thr)

        feat = out_dict["mask_feat"][0]
        decoded_lanes = []

        for s in grouped_seeds:
            x, y = s["coord"]
            mask = dynamic_conv(feat, out_dict["mask_w"][0, :, y, x], out_dict["mask_b"][0, 0, y, x])
            reg  = dynamic_conv(feat, out_dict["reg_w"][0, :, y, x], out_dict["reg_b"][0, 0, y, x])

            span = row_span(lane_range(mask, self.mlp))
            if span is None:
                continue

            e = np.exp(mask - mask.max(1, keepdims=True))
            col = ((e / e.sum(1, keepdims=True)) * _COL).sum(1).astype(np.int32)
            rows = np.arange(span[0], span[1] + 1)
            pts_net = np.stack([col[rows] + reg[rows, col[rows]], rows], 1).astype(np.float32) * 8.0

            if len(pts_net) > 1:
                # Map from network pixels (800x320) back to raw camera frame pixels
                u_frame = rx0 + (pts_net[:, 0] / float(VisionPilotConfig.CONDLANE_NET_W)) * (rx1 - rx0)
                v_frame = ry0 + (pts_net[:, 1] / float(VisionPilotConfig.CONDLANE_NET_H)) * (ry1 - ry0)
                pts_frame = np.stack([u_frame, v_frame], axis=1)

                decoded_lanes.append({
                    "points_net": pts_net,
                    "points_frame": pts_frame,
                    "score": s["score"],
                    "seed": s["coord"]
                })

        # ─── Slot Assignment: Ego-Left, Ego-Right, Left-Outer, Right-Outer ──
        slots = self._assign_slots(decoded_lanes, raw_w, raw_h)

        # ─── Compute ADAS Lane Metrics ──────────────────────────────────────
        lane_metrics = self._compute_lane_metrics(slots, raw_w, raw_h)

        return {
            "valid": len(decoded_lanes) > 0,
            "lanes": decoded_lanes,
            "num_lanes": len(decoded_lanes),
            "slots": slots,
            "metrics": lane_metrics,
            "latency_ms": self.last_latency_ms
        }

    def _assign_slots(self, lanes: list, w: int, h: int) -> dict:
        left, right = [], []
        center_x = w * 0.5

        for lane in lanes:
            pts = lane["points_frame"]
            # Near-field line through bottom-most points
            near = pts[pts[:, 1].argsort()][-12:]
            if len(near) < 2:
                continue
            poly = np.polyfit(near[:, 1], near[:, 0], 1)
            u_bottom = float(np.polyval(poly, h))
            dist_to_center = abs(u_bottom - center_x)
            
            entry = (dist_to_center, lane, pts, u_bottom)
            if u_bottom < center_x:
                left.append(entry)
            else:
                right.append(entry)

        left.sort(key=lambda t: t[0])
        right.sort(key=lambda t: t[0])

        slots = {}
        # Ego-left: id 0, Left-outer: id 2
        for slot_id, entry in zip([0, 2], left):
            slots[slot_id] = {
                "name": self.SLOT_NAMES[slot_id],
                "lane": entry[1],
                "points": entry[2],
                "u_bottom": entry[3],
                "score": entry[1]["score"]
            }
        # Ego-right: id 1, Right-outer: id 3
        for slot_id, entry in zip([1, 3], right):
            slots[slot_id] = {
                "name": self.SLOT_NAMES[slot_id],
                "lane": entry[1],
                "points": entry[2],
                "u_bottom": entry[3],
                "score": entry[1]["score"]
            }

        return slots

    def _compute_lane_metrics(self, slots: dict, w: int, h: int) -> dict:
        ego_left = slots.get(0)
        ego_right = slots.get(1)

        has_ego_lane = (ego_left is not None and ego_right is not None)
        lane_width_px = 0.0
        center_offset_px = 0.0
        ldw_state = "NO_WARNING"

        if has_ego_lane:
            u_l = ego_left["u_bottom"]
            u_r = ego_right["u_bottom"]
            lane_width_px = u_r - u_l
            lane_center_px = (u_l + u_r) * 0.5
            center_offset_px = (w * 0.5) - lane_center_px

            # Approximate lane departure if car center is close to either boundary
            # Standard lane ~ 3.5m, car ~ 1.8m
            if lane_width_px > 100:
                norm_offset = center_offset_px / lane_width_px
                if norm_offset > 0.22:
                    ldw_state = "WARN_RIGHT"
                elif norm_offset < -0.22:
                    ldw_state = "WARN_LEFT"
        elif ego_left is not None:
            if ego_left["u_bottom"] > w * 0.40:
                ldw_state = "WARN_LEFT"
        elif ego_right is not None:
            if ego_right["u_bottom"] < w * 0.60:
                ldw_state = "WARN_RIGHT"

        return {
            "has_ego_lane": has_ego_lane,
            "lane_width_px": float(lane_width_px),
            "center_offset_px": float(center_offset_px),
            "ldw_state": ldw_state,
            "detected_slots": [s["name"] for s in slots.values()]
        }
