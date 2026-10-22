#!/usr/bin/env python3
"""
Unit and Integration Test Suite for VisionPilot INT8 Modules.
Tests preprocessing, model loaders, homography projection, fusion, planning, and HUD rendering.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Ensure path
base_dir = Path(__file__).resolve().parent
if str(base_dir.parent) not in sys.path:
    sys.path.insert(0, str(base_dir.parent))

from int8visionpilot.config.config import VisionPilotConfig
from int8visionpilot.calibration.homography import HomographyTransformer, top_crop_2_1
from int8visionpilot.fusion.lateral_fusion import LateralFusion
from int8visionpilot.fusion.longitudinal_fusion import LongitudinalFusion
from int8visionpilot.planning.planner import VisionPilotPlanner
from int8visionpilot.visualization.hud import VisionPilotHUD

class TestVisionPilotComponents(unittest.TestCase):
    def setUp(self):
        self.homography = HomographyTransformer(
            VisionPilotConfig.DEFAULT_H_MATRIX,
            raw_shape=(1080, 1920),
            net_shape=(512, 1024)
        )
        self.lateral_fusion = LateralFusion(self.homography)
        self.longitudinal_fusion = LongitudinalFusion(self.homography)
        self.planner = VisionPilotPlanner()
        self.hud = VisionPilotHUD(out_width=1920, out_height=1080)

    def test_top_crop(self):
        crop = top_crop_2_1(1080, 1920)
        self.assertEqual(crop, 120)  # (1080 - 1920 / 2) = 1080 - 960 = 120

    def test_homography_projections(self):
        # Center bottom of model space (512, 511)
        x_fwd, y_lat = self.homography.img_to_world(512.0, 500.0, is_model_space=True)
        self.assertTrue(x_fwd > 0.0)
        # Project back
        u, v = self.homography.world_to_img(x_fwd, y_lat, to_model_space=True)
        self.assertAlmostEqual(u, 512, delta=3)
        self.assertAlmostEqual(v, 500, delta=3)

    def test_lateral_fusion(self):
        fake_steer = {
            "model_pts": [(512.0 + i * 2, 500.0 - i * 6) for i in range(20)],
            "num_valid_waypoints": 20
        }
        fake_drive = {
            "valid": True,
            "curvature": 0.002
        }
        res = self.lateral_fusion.update(fake_steer, fake_drive, dt=0.033)
        self.assertTrue(res["valid"])
        self.assertIn("cte_m", res)
        self.assertIn("curvature", res)

    def test_longitudinal_fusion(self):
        fake_speed = {
            "detections": [{
                "x1": 480.0, "y1": 300.0, "x2": 540.0, "y2": 360.0,
                "score": 0.85, "class_id": 1, "class_name": "CIPO-1 (Lead Vehicle)",
                "bottom_center": (510.0, 360.0)
            }]
        }
        fake_drive = {
            "valid": True,
            "distance_m": 35.0,
            "flag_prob": 0.92,
            "has_obstacle": True
        }
        res = self.longitudinal_fusion.update(fake_drive, fake_speed, dt=0.033)
        self.assertTrue(res["valid"])
        self.assertTrue(res["has_cipo"])
        self.assertTrue(0.0 < res["distance_m"] < 150.0)

    def test_planner_normal_and_aeb(self):
        lat = {"cte_m": 0.1, "yaw_rad": 0.02, "curvature": 0.001}
        
        # Test normal following
        cipo_normal = {"has_cipo": True, "distance_m": 45.0, "velocity_ms": 0.0, "ttc_s": 99.9}
        plan_normal = self.planner.compute_plan(lat, cipo_normal, ego_v_mps=15.0)
        self.assertEqual(plan_normal["aeb_state"], "NORMAL")
        self.assertFalse(plan_normal["aeb_active"])
        self.assertTrue(-2.0 <= plan_normal["acceleration_mps2"] <= 2.5)

        # Test critical collision approach (high closing speed, small distance -> TTC < 1.0s)
        cipo_crit = {"has_cipo": True, "distance_m": 12.0, "velocity_ms": -14.0, "ttc_s": 0.85}
        plan_crit = self.planner.compute_plan(lat, cipo_crit, ego_v_mps=15.0)
        self.assertEqual(plan_crit["aeb_state"], "AEB_STAGE2_EMERGENCY")
        self.assertTrue(plan_crit["aeb_active"])
        self.assertEqual(plan_crit["acceleration_mps2"], -8.0)

    def test_condlane_lateral_fusion(self):
        fake_steer = {"model_pts": [(512, 500)], "num_valid_waypoints": 1}
        fake_drive = {"valid": False}
        fake_condlane = {
            "valid": True,
            "metrics": {
                "has_ego_lane": True,
                "lane_width_px": 350.0,
                "ldw_state": "WARN_LEFT"
            },
            "slots": {
                0: {"name": "ego-left", "points": np.array([(400, 700), (450, 900)])},
                1: {"name": "ego-right", "points": np.array([(1400, 700), (1450, 900)])}
            }
        }
        res = self.lateral_fusion.update(fake_steer, fake_drive, condlane_output=fake_condlane, dt=0.033)
        self.assertTrue(res["valid"])
        self.assertTrue(res["has_condlane"])
        self.assertEqual(res["ldw_state"], "WARN_LEFT")

    def test_hud_render(self):
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        drive_res = {"valid": True, "latency_ms": 21.5, "distance_m": 25.0, "flag_prob": 0.85}
        steer_res = {"valid": True, "latency_ms": 14.2, "model_pts": [(512, 500), (514, 450), (518, 400)]}
        speed_res = {"valid": True, "latency_ms": 17.8, "detections": [{
            "x1": 480, "y1": 300, "x2": 540, "y2": 360, "score": 0.9,
            "class_name": "CIPO-1", "class_id": 1, "bottom_center": (510, 360), "x_fwd": 25.0
        }]}
        condlane_res = {
            "valid": True, "latency_ms": 12.3,
            "slots": {
                0: {"name": "ego-left", "points": np.array([[400, 700], [450, 900]])},
                1: {"name": "ego-right", "points": np.array([[1400, 700], [1450, 900]])}
            }
        }
        lat_est = {"cte_m": 0.05, "curvature": 0.0005, "yaw_rad": 0.01, "has_condlane": True, "ldw_state": "NOMINAL", "lane_width_m": 3.5}
        cipo_est = {"has_cipo": True, "distance_m": 25.0, "velocity_ms": -5.0, "ttc_s": 5.0, "best_cipo": speed_res["detections"][0]}
        plan = {"aeb_state": "NORMAL", "aeb_active": False, "steering_angle_deg": 1.2, "acceleration_mps2": 0.5, "ttc_s": 5.0}

        out = self.hud.render_frame(
            dummy_frame, drive_res, steer_res, speed_res,
            lat_est, cipo_est, plan, ego_speed_mps=15.0, frame_idx=10,
            condlane_res=condlane_res, video_name="TestVideo"
        )
        self.assertEqual(out.shape, (1080, 1920, 3))

if __name__ == "__main__":
    unittest.main()
