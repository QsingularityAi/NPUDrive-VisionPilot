"""AutoSpeed Model: Obstacle & Lead Vehicle Perception (YOLO Detector)."""
import cv2
import numpy as np
from .base_model import BaseTIDLModel
from ..config.config import VisionPilotConfig
from ..calibration.homography import top_crop_2_1

def _init_static_anchors():
    strides = [8.0, 16.0, 32.0]
    shapes = [(VisionPilotConfig.NET_H // int(s), VisionPilotConfig.NET_W // int(s)) for s in strides]
    anchors_list, strides_list = [], []
    for s, (h, w) in zip(strides, shapes):
        sy, sx = np.meshgrid(
            np.arange(h, dtype=np.float32) + 0.5,
            np.arange(w, dtype=np.float32) + 0.5,
            indexing="ij"
        )
        grid = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        anchors_list.append(grid)
        strides_list.append(np.full((h * w, 1), s, dtype=np.float32))
    return np.concatenate(anchors_list, axis=0), np.concatenate(strides_list, axis=0)

_STATIC_ANCHORS, _STATIC_STRIDES = _init_static_anchors()

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    x1 = boxes[:, 0] - boxes[:, 2] * 0.5
    y1 = boxes[:, 1] - boxes[:, 3] * 0.5
    x2 = boxes[:, 0] + boxes[:, 2] * 0.5
    y2 = boxes[:, 1] + boxes[:, 3] * 0.5
    return np.stack([x1, y1, x2, y2], axis=-1)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> list:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep

class AutoSpeedModel(BaseTIDLModel):
    CLASS_NAMES = {0: "Obstacle", 1: "CIPO-1 (Lead Vehicle)", 2: "CIPO-2 (Cut-In)", 3: "CIPO-3 (Adjacent)"}

    def __init__(self, onnx_path: str, artifacts_dir: str, prefer_tidl: bool = True):
        super().__init__("autospeed", onnx_path, artifacts_dir, prefer_tidl=prefer_tidl)

    def preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        """Top-crops frame to 2:1 aspect ratio, resizes to 1024x512, normalizes RGB to [0, 1]."""
        h, w = bgr_img.shape[:2]
        crop_top = top_crop_2_1(h, w)
        cropped = bgr_img[crop_top:, :]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (VisionPilotConfig.NET_W, VisionPilotConfig.NET_H), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

    def infer(self, curr_bgr: np.ndarray, conf_thres: float = None, iou_thres: float = None) -> dict:
        if conf_thres is None: conf_thres = VisionPilotConfig.AUTOSPEED_CONF_THRES
        if iou_thres is None: iou_thres = VisionPilotConfig.AUTOSPEED_IOU_THRES

        inp = self.preprocess(curr_bgr)
        outs = self.run({self.input_names[0]: inp})
        if len(outs) == 1:
            raw_out = outs[0]  # shape (1, 8, 10752)
            raw_boxes = raw_out[0, :4, :]       # (4, 10752)
            raw_scores = raw_out[0, 4:, :]      # (4, 10752)
            boxes = raw_boxes.T                 # (10752, 4)
            scores = raw_scores.T
        else:
            # Dual output (v2 decoupled architecture)
            ltrb = outs[0]
            scores = outs[1][0].T               # (10752, 4)
            l = ltrb[0, 0, :]
            t = ltrb[0, 1, :]
            r = ltrb[0, 2, :]
            b = ltrb[0, 3, :]
            cx = (_STATIC_ANCHORS[:, 0] + (r - l) * 0.5) * _STATIC_STRIDES[:, 0]
            cy = (_STATIC_ANCHORS[:, 1] + (b - t) * 0.5) * _STATIC_STRIDES[:, 0]
            w = (r + l) * _STATIC_STRIDES[:, 0]
            h = (b + t) * _STATIC_STRIDES[:, 0]
            boxes = np.stack([cx, cy, w, h], axis=-1)

        # Sigmoid on logits if necessary (or raw if model output already probabilities)
        if np.max(scores) > 1.0 or np.min(scores) < 0.0:
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = scores

        # Find best class per candidate anchor
        class_ids = np.argmax(probs, axis=1)
        max_scores = np.max(probs, axis=1)

        mask = max_scores >= conf_thres
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]

        detections = []
        if len(filtered_boxes) > 0:
            xyxy_boxes = xywh2xyxy(filtered_boxes)
            # Clip to model bounds
            xyxy_boxes[:, 0] = np.clip(xyxy_boxes[:, 0], 0, VisionPilotConfig.NET_W)
            xyxy_boxes[:, 1] = np.clip(xyxy_boxes[:, 1], 0, VisionPilotConfig.NET_H)
            xyxy_boxes[:, 2] = np.clip(xyxy_boxes[:, 2], 0, VisionPilotConfig.NET_W)
            xyxy_boxes[:, 3] = np.clip(xyxy_boxes[:, 3], 0, VisionPilotConfig.NET_H)

            keep_idx = nms(xyxy_boxes, filtered_scores, iou_thres=iou_thres)
            for idx in keep_idx:
                box = xyxy_boxes[idx]
                cid = int(filtered_classes[idx])
                detections.append({
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "score": float(filtered_scores[idx]),
                    "class_id": cid,
                    "class_name": self.CLASS_NAMES.get(cid, "Obstacle"),
                    "bottom_center": ((box[0] + box[2]) * 0.5, float(box[3]))
                })

        return {
            "valid": len(detections) > 0,
            "detections": detections,
            "num_detections": len(detections),
            "latency_ms": self.last_latency_ms
        }
