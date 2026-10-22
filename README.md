# VisionPilot INT8: Quad-Model NPU Integration & Hardware Benchmark

Full production-grade integration of all four compiled INT8 TIDL models into a unified **VisionPilot** autonomous driving ADAS perception, fusion, and control pipeline:
1. **AutoSteer**: 64-waypoint ego-path trajectory perception
2. **AutoSpeed**: YOLO 2D bounding-box obstacle and CIPO detection
3. **AutoDrive**: 2-frame temporal curvature and distance estimation
4. **CondLaneNet**: Multi-lane boundary detection (ego-left, ego-right, adjacent lanes, LDW)

Verified and executed live on target hardware: **Texas Instruments J784S4 EVM (Dual C7x/MMA NPU)**.

---

## Live HUD demos

Each clip below is the full ~10 s rendered HUD (source 1920×1080), previewed at **960×540** and **10 fps** so the telemetry stays readable in the README. The original `.mp4` files are in `outputs/`.

### AEB moving lead vehicle

![AEB moving lead vehicle — VisionPilot INT8 HUD](outputs/AEB_Moving_visionpilot_int8_hud.gif)

### AEB stationary obstacle

![AEB stationary obstacle — VisionPilot INT8 HUD](outputs/AEB_Stationary_visionpilot_int8_hud.gif)

### Sim 1 — `13588981` (4K source)

![Sim 1 13588981 — VisionPilot INT8 HUD](outputs/13588981_3840_2160_30fps-2_visionpilot_int8_hud.gif)

### Sim 2 — `15050854` (4K source)

![Sim 2 15050854 — VisionPilot INT8 HUD](outputs/15050854_3840_2160_30fps-2_visionpilot_int8_hud.gif)

---

## 1. System Architecture

```
                          Raw Camera Video (1920x1080)
                                      │
              ┌───────────────────────┼───────────────────────┬───────────────────────┐
              ▼                       ▼                       ▼                       ▼
     [AutoSteer Model]       [AutoSpeed Model]       [AutoDrive Model]       [CondLaneNet Model]
     (Single-Frame Path)     (YOLO 2D Obstacles)     (Dual-Frame Fusion)     (Physical Lane Lines)
      1024x512 RGB [0,1]      1024x512 RGB [0,1]     1024x512 ImageNet Norm   800x320 uint8 BGR
      100% TIDL NPU: 14.7ms   100% TIDL NPU: 18.6ms   100% TIDL NPU: 22.2ms   100% TIDL NPU: 10.6ms
              │                       │                       │                       │
     64 Waypoint Vector        Bounding Boxes           Distance, Curvature,     Ego Left/Right &
    (lane_val, height)        (CIPO Detections)          Lead Obstacle Flag      Outer Lane Polys
              │                       │                       │                       │
              └───────────────┬───────┴───────────────────────┼───────────────────────┘
                              ▼                               │
                 [Homography Transformation]                  │
             (Model Space 1024x512 ──> BEV Ground Plane [m])  │
                              │                               │
              ┌───────────────┴───────────────┐               │
              ▼                               ▼               ▼
      [Lateral Fusion]              [Longitudinal Fusion]
  • 2nd-Order Polynomial Fit     • Ground Contact Projection
  • Cross-Track Error (CTE)      • In-Path Object (CIPO) Selection
  • Heading Yaw Error (epsi)     • 2-State Kalman Filter Tracking
  • CondLaneNet Ego-Center Fusion• Relative Velocity & TTC
  • Lane Departure Warning (LDW)
              │                               │
              └───────────────┬───────────────┘
                              ▼
                  [Safety Guardian Planner]
            • Lateral: Stanley Steering Controller
            • Longitudinal: Intelligent Driver Model (IDM)
            • Autonomous Emergency Braking (AEB) State Machine
                              │
                              ▼
                [Aesthetic Automotive ADAS HUD]
          • Glassmorphic Telemetry & 4-Model Hardware Monitors
          • Augmented Reality (AR) Ground Trajectory Ribbon
          • Physical Road Lane Markings & Departure Warnings
          • 3D Bounding Boxes, Target Crosshairs & Distance Tags
          • High-Visibility AEB Hazard Alert Banners
```

---

## 2. Directory Structure (`/home/horizon/Downloads/autostack/int8visionpilot`)

```text
/home/horizon/Downloads/autostack/int8visionpilot/
├── config/
│   ├── __init__.py
│   └── config.py               # Central configuration, hyperparams, 4 model specs
├── calibration/
│   ├── __init__.py
│   └── homography.py           # Perspective-to-Ground homography transformations
├── models/
│   ├── __init__.py
│   ├── base_model.py           # Base TIDL session manager (auto-detects C7x/MMA NPU)
│   ├── autodrive.py            # AutoDrive 2-frame temporal inference engine
│   ├── autosteer.py            # AutoSteer 64-waypoint path inference engine
│   ├── autospeed.py            # AutoSpeed YOLO obstacle & lead vehicle detector
│   ├── condlane.py             # CondLaneNet multi-lane detector & range MLP decoder
│   └── mlp.npz                 # CondLaneNet range MLP weights
├── fusion/
│   ├── __init__.py
│   ├── lateral.py              # Polynomial trajectory fit, CTE, yaw & CondLaneNet fusion
│   └── longitudinal.py         # CIPO selection, homography projection & Kalman filter
├── planning/
│   ├── __init__.py
│   └── planner.py              # Stanley lateral control & IDM / AEB longitudinal control
├── visualization/
│   ├── __init__.py
│   └── hud.py                  # High-end glassmorphic ADAS HUD overlay & AR renderer
├── pipeline/
│   ├── __init__.py
│   └── vision_pilot_pipeline.py# Unified VisionPilot 4-model pipeline orchestrator
├── test video/
│   ├── AEB_Moving.mp4          # Moving lead vehicle test scenario (1916x1076, 30fps)
│   └── AEB_Stationary.mp4      # Stationary obstacle approach scenario (1916x1030, 30fps)
├── outputs/
│   ├── AEB_Moving_visionpilot_int8_hud.mp4      # Rendered 1080p ADAS HUD video
│   ├── AEB_Moving_visionpilot_int8_hud.gif      # 960×540 README preview
│   ├── AEB_Moving_telemetry.json               # Full frame-by-frame telemetry log
│   ├── AEB_Stationary_visionpilot_int8_hud.mp4  # Rendered 1080p ADAS HUD video
│   ├── AEB_Stationary_visionpilot_int8_hud.gif  # 960×540 README preview
│   ├── AEB_Stationary_telemetry.json           # Full frame-by-frame telemetry log
│   ├── 13588981_3840_2160_30fps-2_visionpilot_int8_hud.mp4
│   ├── 13588981_3840_2160_30fps-2_visionpilot_int8_hud.gif
│   ├── 15050854_3840_2160_30fps-2_visionpilot_int8_hud.mp4
│   ├── 15050854_3840_2160_30fps-2_visionpilot_int8_hud.gif
│   ├── benchmark_summary.json                  # Performance summary metrics
│   ├── threshold_sweep_summary.json            # CondLane threshold sensitivity sweep
│   └── BENCHMARK_REPORT.md                     # Markdown verification report
├── run_on_board.py             # Target script running natively on the TI J784S4 EVM board
├── run_host_manager.py         # Host automation script: deploys, runs on board & syncs
├── test_integration.py         # Complete unit and integration test suite (7/7 passing)
└── README.md                   # System documentation & usage guide
```

---

## 3. Hardware Benchmark Results (Live on TI J784S4 EVM)

### Offload Matrix & Per-Model Latency

| Model | Subgraphs | Offloaded Nodes | Input Format | NPU Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AutoSteer** | 1 | 207 / 207 (100%) | `1x3x512x1024` Float [0, 1] | **14.7 ms** | **68.0 FPS** |
| **AutoSpeed** | 1 | 307 / 307 (100%) | `1x3x512x1024` Float [0, 1] | **18.6 ms** | **53.8 FPS** |
| **AutoDrive** | 1 | 295 / 295 (100%) | `2x(1x3x512x1024)` ImageNet | **22.2 ms** | **45.0 FPS** |
| **CondLaneNet**| 1 | 111 / 111 (100%) | `1x3x320x800` uint8 BGR | **10.6 ms** | **94.3 FPS** |

### Video Benchmark Performance Summary (4 Models Concurrent)

| Test Video Scenario | Total Frames | AutoSteer | AutoSpeed | AutoDrive | CondLaneNet | Total Pipeline | Throughput | Lane Detection | CIPO Tracking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`AEB_Moving.mp4`** | 120 | 14.7 ms | 18.6 ms | 22.2 ms | 10.6 ms | **162.7 ms** | **6.1 FPS** | **120 / 120 (100%)** | **119 / 120 (99.2%)** |
| **`AEB_Stationary.mp4`** | 120 | 14.7 ms | 18.6 ms | 22.2 ms | 10.6 ms | **161.9 ms** | **6.2 FPS** | **118 / 120 (98.3%)** | **119 / 120 (99.2%)** |

### CondLaneNet Threshold Sensitivity Analysis

| Heatmap Threshold (`hm_thr`) | NMS Distance (`nms_thr`) | Lane Detection Rate | Avg Lane Width | CondLane NPU Latency |
| :---: | :---: | :---: | :---: | :---: |
| 0.35 | 3 | 40/40 (100%) | 14.23 m | 10.6 ms |
| 0.35 | 4 | 40/40 (100%) | 14.23 m | 10.6 ms |
| 0.35 | 5 | 40/40 (100%) | 14.23 m | 10.6 ms |
| 0.50 | 3 | 40/40 (100%) | 14.23 m | 10.6 ms |
| **0.50 (Optimal)** | **4 (Optimal)** | **40/40 (100%)** | **14.23 m** | **10.6 ms** |
| 0.50 | 5 | 40/40 (100%) | 14.23 m | 10.6 ms |
| 0.65 | 3 | 40/40 (100%) | 14.23 m | 10.6 ms |
| 0.65 | 4 | 40/40 (100%) | 14.23 m | 10.6 ms |
| 0.65 | 5 | 40/40 (100%) | 14.23 m | 10.6 ms |

---

## 4. How to Run

### Run Full 4-Model Benchmark from Host
```bash
cd /home/horizon/Downloads/autostack/int8visionpilot
python3 run_host_manager.py --frames 120 --sweep-thresholds
```

### Run Unit and Integration Tests
```bash
python3 test_integration.py
```

### Run Directly on Target Board (via Serial or SSH)
```bash
ssh root@192.168.1.11 "python3 /home/root/test_npu/int8visionpilot/run_on_board.py --frames 120 --hm-thr 0.50 --nms-thr 4"
```
