"""Unified VisionPilot INT8 Pipeline: Perception -> Fusion -> Planning -> HUD."""
import time
import numpy as np
from ..config.config import VisionPilotConfig
from ..calibration.homography import HomographyTransformer
from ..models.autodrive import AutoDriveModel
from ..models.autosteer import AutoSteerModel
from ..models.autospeed import AutoSpeedModel
from ..models.condlane import CondLaneModel
from ..fusion.lateral_fusion import LateralFusion
from ..fusion.longitudinal_fusion import LongitudinalFusion
from ..planning.planner import VisionPilotPlanner
from ..visualization.hud import VisionPilotHUD

class VisionPilotPipeline:
    def __init__(self, model_configs: dict = None, prefer_tidl: bool = True):
        cfg = model_configs if model_configs else VisionPilotConfig.BOARD_MODELS
        
        print("================================================================================")
        print("          INITIALIZING VISIONPILOT QUAD INT8 TIDL PIPELINE (4 MODELS)           ")
        print("================================================================================")

        # 1. Initialize 4 Compiled INT8 Models
        self.autodrive = AutoDriveModel(
            cfg["autodrive"]["onnx"], cfg["autodrive"]["artifacts"], prefer_tidl=prefer_tidl
        )
        self.autosteer = AutoSteerModel(
            cfg["autosteer"]["onnx"], cfg["autosteer"]["artifacts"], prefer_tidl=prefer_tidl
        )
        self.autospeed = AutoSpeedModel(
            cfg["autospeed"]["onnx"], cfg["autospeed"]["artifacts"], prefer_tidl=prefer_tidl
        )
        self.condlane = None
        if "condlane" in cfg:
            self.condlane = CondLaneModel(
                cfg["condlane"]["onnx"], cfg["condlane"]["artifacts"],
                cfg["condlane"].get("mlp", str(VisionPilotConfig.HOST_ROOT / "condlane_tidl" / "work" / "models" / "mlp.npz")),
                prefer_tidl=prefer_tidl
            )

        # 2. Homography Transformer
        self.homography = HomographyTransformer(
            VisionPilotConfig.DEFAULT_H_MATRIX,
            raw_shape=(1080, 1920),
            net_shape=(VisionPilotConfig.NET_H, VisionPilotConfig.NET_W)
        )

        # 3. Safety Guardian: Fusion
        self.lateral_fusion = LateralFusion(self.homography)
        self.longitudinal_fusion = LongitudinalFusion(self.homography)

        # 4. Safety Guardian: Planning
        self.planner = VisionPilotPlanner()

        # 5. Aesthetics HUD
        self.hud = VisionPilotHUD(out_width=1920, out_height=1080)

        # Stats
        self.frame_count = 0
        self.total_pipeline_time_ms = 0.0

        print("================================================================================")
        print("          VISIONPILOT INT8 PIPELINE READY FOR HARDWARE INFERENCE                ")
        print("================================================================================\n")

    def process_frame(self, raw_bgr_frame: np.ndarray,
                      ego_speed_mps: float = 18.0,
                      frame_idx: int = 0,
                      video_name: str = "Video",
                      dt: float = 0.033,
                      hm_thr: float = None,
                      nms_thr: int = None,
                      render_hud: bool = True) -> tuple:
        
        t0 = time.time()
        
        # ─── 1. Run 4 Compiled INT8 Models ──────────────────────────────────
        drive_res = self.autodrive.infer(raw_bgr_frame)
        steer_res = self.autosteer.infer(raw_bgr_frame)
        speed_res = self.autospeed.infer(raw_bgr_frame)
        
        condlane_res = None
        if self.condlane is not None:
            condlane_res = self.condlane.infer(raw_bgr_frame, hm_thr=hm_thr, nms_thr=nms_thr)

        # ─── 2. Sensor Fusion ───────────────────────────────────────────────
        lat_est = self.lateral_fusion.update(steer_res, drive_res, condlane_output=condlane_res, dt=dt)
        cipo_est = self.longitudinal_fusion.update(drive_res, speed_res, dt=dt)

        # ─── 3. Control Planning ────────────────────────────────────────────
        plan = self.planner.compute_plan(lat_est, cipo_est, ego_speed_mps)

        total_ms = (time.time() - t0) * 1000.0
        self.frame_count += 1
        self.total_pipeline_time_ms += total_ms

        # ─── 4. Render HUD Display Frame ─────────────────────────────────────
        hud_frame = None
        if render_hud:
            hud_frame = self.hud.render_frame(
                raw_bgr_frame,
                drive_res, steer_res, speed_res,
                lat_est, cipo_est, plan,
                ego_speed_mps, frame_idx,
                condlane_res=condlane_res,
                video_name=video_name
            )

        telemetry = {
            "frame_idx": frame_idx,
            "total_pipeline_ms": total_ms,
            "drive_res": drive_res,
            "steer_res": steer_res,
            "speed_res": speed_res,
            "condlane_res": condlane_res,
            "lat_est": lat_est,
            "cipo_est": cipo_est,
            "plan": plan,
            "ego_speed_mps": ego_speed_mps
        }

        return hud_frame, telemetry

    def reset(self):
        self.autodrive.reset()
        self.lateral_fusion.reset()
        self.longitudinal_fusion.reset()
        self.frame_count = 0
