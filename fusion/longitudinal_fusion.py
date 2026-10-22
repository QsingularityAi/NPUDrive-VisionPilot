"""Longitudinal Fusion: CIPO Selection, Homography Projection & Kalman Tracking."""
import numpy as np
from ..calibration.homography import HomographyTransformer
from ..config.config import VisionPilotConfig

class LongitudinalFusion:
    def __init__(self, homography: HomographyTransformer):
        self.homography = homography
        
        # 1D Kalman filter for distance & relative velocity: [dist, vel]
        self.x = np.array([VisionPilotConfig.AUTODRIVE_D_MAX_M, 0.0], dtype=np.float64)
        self.P = np.diag([25.0, 4.0]).astype(np.float64)
        self.Q = np.diag([0.25, 0.50]).astype(np.float64)  # Process noise
        self.R = 4.0                                       # Measurement noise
        
        self.has_active_track = False
        self.missed_frames = 0
        self.max_missed_frames = 15

    def update(self, drive_output: dict, speed_output: dict, dt: float = 0.033) -> dict:
        detections = speed_output.get("detections", [])
        
        # 1. Select CIPO from AutoSpeed detections
        best_cipo = None
        min_dist = 999.0
        cipo_candidates = []

        for det in detections:
            u_bc, v_bc = det["bottom_center"]
            x_fwd, y_lat = self.homography.img_to_world(u_bc, v_bc, is_model_space=True)
            
            # Ground projection sanity check: forward distance > 0m
            if 1.0 <= x_fwd <= 150.0:
                det_info = {
                    **det,
                    "x_fwd": float(x_fwd),
                    "y_lat": float(y_lat)
                }
                cipo_candidates.append(det_info)
                
                # Check if inside ego lane corridor (|y_lat| <= 2.2m) or classified as Lead (CIPO-1/2)
                in_corridor = abs(y_lat) <= 2.2 or det["class_id"] in [1, 2]
                if in_corridor and x_fwd < min_dist:
                    min_dist = x_fwd
                    best_cipo = det_info

        # 2. Evaluate AutoDrive measurement
        ad_dist = drive_output.get("distance_m", VisionPilotConfig.AUTODRIVE_D_MAX_M)
        ad_flag_prob = drive_output.get("flag_prob", 0.0)
        ad_has_obstacle = drive_output.get("has_obstacle", False)

        # 3. Sensor Fusion Measurement Formulation
        meas_dist = None
        meas_weight = 0.0

        if best_cipo is not None and ad_has_obstacle:
            # Both detected obstacle -> Optimal weighted fusion
            meas_dist = 0.70 * best_cipo["x_fwd"] + 0.30 * ad_dist
            meas_weight = 1.0
        elif best_cipo is not None:
            # AutoSpeed detected -> use homography distance
            meas_dist = best_cipo["x_fwd"]
            meas_weight = 0.85
        elif ad_has_obstacle and ad_dist < 100.0:
            # AutoDrive detected
            meas_dist = ad_dist
            meas_weight = 0.60

        # 4. Kalman Filter Predict & Update
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        
        if meas_dist is not None:
            if not self.has_active_track:
                # Initialize tracker
                self.x = np.array([meas_dist, 0.0], dtype=np.float64)
                self.P = np.diag([4.0, 2.0]).astype(np.float64)
                self.has_active_track = True
            else:
                # Predict
                self.x = np.matmul(F, self.x)
                self.P = np.matmul(np.matmul(F, self.P), F.T) + self.Q
                
                # Update
                H_mat = np.array([[1.0, 0.0]], dtype=np.float64)
                y_innov = meas_dist - self.x[0]
                S = float(np.matmul(np.matmul(H_mat, self.P), H_mat.T) + self.R / meas_weight)
                K = (np.matmul(self.P, H_mat.T) / S).flatten()
                
                self.x += K * y_innov
                self.P = self.P - np.outer(K, np.matmul(H_mat, self.P))

            self.missed_frames = 0
        else:
            if self.has_active_track:
                # Coast with constant velocity
                self.x = np.matmul(F, self.x)
                self.P = np.matmul(np.matmul(F, self.P), F.T) + self.Q
                self.missed_frames += 1
                if self.missed_frames > self.max_missed_frames:
                    self.has_active_track = False
                    self.x = np.array([VisionPilotConfig.AUTODRIVE_D_MAX_M, 0.0])

        fused_dist = float(np.clip(self.x[0], 0.5, VisionPilotConfig.AUTODRIVE_D_MAX_M))
        fused_vel = float(self.x[1])  # Negative = closing in

        # 5. Compute Time-To-Collision (TTC)
        # When fused_vel < -0.3 m/s, target is closing in
        if fused_vel < -0.3 and fused_dist < 120.0:
            ttc_s = float(fused_dist / abs(fused_vel))
        else:
            ttc_s = 99.9

        return {
            "valid": self.has_active_track,
            "has_cipo": self.has_active_track and fused_dist < VisionPilotConfig.AUTODRIVE_D_MAX_M - 5.0,
            "distance_m": fused_dist,
            "velocity_ms": fused_vel,
            "ttc_s": ttc_s,
            "best_cipo": best_cipo,
            "candidates": cipo_candidates,
            "raw_meas_dist": meas_dist
        }

    def reset(self):
        self.x = np.array([VisionPilotConfig.AUTODRIVE_D_MAX_M, 0.0], dtype=np.float64)
        self.P = np.diag([25.0, 4.0]).astype(np.float64)
        self.has_active_track = False
        self.missed_frames = 0
