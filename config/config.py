"""VisionPilot INT8 Configuration & Hyperparameters."""
import os
from pathlib import Path
import numpy as np

class VisionPilotConfig:
    # ─── Neural Network Dimensions ──────────────────────────────────────────
    NET_W = 1024
    NET_H = 512
    CHW_SIZE = 3 * NET_H * NET_W

    # Preprocessing constants (ImageNet for AutoDrive)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    # ─── Model Paths on Target Board (TI J784S4 EVM) ─────────────────────────
    BOARD_ROOT = Path("/home/root/test_npu")
    BOARD_MODELS = {
        "autodrive": {
            "onnx": str(BOARD_ROOT / "autodrive" / "autodrive_npu.onnx"),
            "artifacts": str(BOARD_ROOT / "autodrive" / "artifacts"),
            "inputs": ["image_prev", "image_curr"]
        },
        "autosteer": {
            "onnx": str(BOARD_ROOT / "autosteer" / "autosteer_npu.onnx"),
            "artifacts": str(BOARD_ROOT / "autosteer" / "artifacts"),
            "inputs": ["input"]
        },
        "autospeed": {
            "onnx": str(BOARD_ROOT / "autospeed" / "autospeed_npu.onnx"),
            "artifacts": str(BOARD_ROOT / "autospeed" / "artifacts"),
            "inputs": ["input"]
        },
        "condlane": {
            "onnx": str(BOARD_ROOT / "condlane" / "condlane_small_npu.onnx"),
            "artifacts": str(BOARD_ROOT / "condlane" / "artifacts"),
            "mlp": str(BOARD_ROOT / "condlane" / "mlp.npz"),
            "inputs": ["input_u8"]
        }
    }

    # ─── Local Host Fallback Paths (for local testing/debugging) ────────────
    HOST_ROOT = Path("/home/horizon/Downloads/autostack")
    HOST_MODELS = {
        "autodrive": {
            "onnx": str(HOST_ROOT / "autodrive_tidl" / "work" / "models" / "autodrive_npu.onnx"),
            "artifacts": str(HOST_ROOT / "autodrive_tidl" / "work" / "artifacts" / "int8"),
            "inputs": ["image_prev", "image_curr"]
        },
        "autosteer": {
            "onnx": str(HOST_ROOT / "autosteer_tidl" / "work" / "models" / "autosteer_npu.onnx"),
            "artifacts": str(HOST_ROOT / "autosteer_tidl" / "work" / "artifacts" / "int8"),
            "inputs": ["input"]
        },
        "autospeed": {
            "onnx": str(HOST_ROOT / "autospeed_tidl" / "work" / "models" / "autospeed_npu.onnx"),
            "artifacts": str(HOST_ROOT / "autospeed_tidl" / "work" / "artifacts" / "int8"),
            "inputs": ["input"]
        },
        "condlane": {
            "onnx": str(HOST_ROOT / "condlane_tidl" / "work" / "models" / "condlane_small_npu.onnx"),
            "artifacts": str(HOST_ROOT / "condlane_tidl" / "work" / "artifacts" / "int8"),
            "mlp": str(HOST_ROOT / "condlane_tidl" / "work" / "models" / "mlp.npz"),
            "inputs": ["input_u8"]
        }
    }

    # ─── Model Thresholds & Parameters ──────────────────────────────────────
    AUTODRIVE_D_MAX_M = 150.0
    AUTODRIVE_FLAG_THRES = 0.50
    AUTODRIVE_CURV_SCALE = 0.21

    AUTOSTEER_NUM_WAYPOINTS = 64
    AUTOSTEER_CONF_THRES = 0.50

    AUTOSPEED_CONF_THRES = 0.40
    AUTOSPEED_IOU_THRES  = 0.45

    CONDLANE_NET_W = 800
    CONDLANE_NET_H = 320
    CONDLANE_HM_THR = 0.50
    CONDLANE_NMS_THR = 4
    CONDLANE_ROI = (0.00, 0.25, 1.00, 0.75)  # Normalized [x0, y0, x1, y1]

    # ─── Homography Transform ───────────────────────────────────────────────
    # OpenLane 1920x1080 to Ground Plane (BEV coordinates in meters)
    DEFAULT_H_MATRIX = np.array([
        [ 6.044144446030150e-04, -2.459315932924031e-03, -6.056269548874612e+00],
        [ 3.292803725663312e-03,  6.577304929754011e-05, -3.193473627885498e+00],
        [ 1.679596119965616e-05, -1.571346654544380e-03,  1.000000000000000e+00]
    ], dtype=np.float64)

    # ─── Vehicle Parameters ─────────────────────────────────────────────────
    WHEELBASE_M = 2.70      # Front to rear axle distance [m]
    SPEED_LIMIT_MPS = 25.0  # Max cruising speed 90 km/h [m/s]
    MU = 0.85               # Road-tire friction coefficient
    GRAVITY = 9.81          # m/s^2

    # ─── Longitudinal Control (IDM ACC / AEB) ───────────────────────────────
    IDM_S0 = 3.5            # Jam distance gap [m]
    IDM_T = 1.4             # Safe time headway [s]
    IDM_A_MAX = 2.0         # Max comfortable acceleration [m/s^2]
    IDM_B_COMFORT = 2.5     # Comfortable deceleration [m/s^2]
    IDM_DELTA = 4.0         # Acceleration exponent

    # Autonomous Emergency Braking (AEB) thresholds
    AEB_FCW_TTC_S = 2.6     # Forward Collision Warning threshold (TTC < 2.6s)
    AEB_STAGE1_TTC_S = 1.8  # Partial braking warning (TTC < 1.8s)
    AEB_STAGE2_TTC_S = 1.2  # Full emergency braking (TTC < 1.2s, a = -8.0 m/s^2)
    AEB_EMERGENCY_ACCEL = -8.0

    # ─── Target Hardware & Connection ───────────────────────────────────────
    BOARD_IP = "192.168.1.11"
    BOARD_USER = "root"
    SERIAL_PORT = "/dev/ttyUSB0"
    SERIAL_BAUD = 115200
