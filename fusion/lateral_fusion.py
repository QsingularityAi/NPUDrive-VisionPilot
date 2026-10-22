"""Lateral Fusion: Waypoint Projection, Curve Fitting & Tracking State."""
import numpy as np
from ..calibration.homography import HomographyTransformer

class LateralFusion:
    def __init__(self, homography: HomographyTransformer):
        self.homography = homography
        
        # State tracking: [cte, cte_dot, yaw, yaw_dot, curvature]
        self.cte_m = 0.0
        self.cte_rate_mps = 0.0
        self.yaw_rad = 0.0
        self.yaw_rate_rps = 0.0
        self.fused_curvature = 0.0
        
        self.prev_time = None
        self.alpha_state = 0.35  # Smoothing factor
        self.alpha_curv = 0.30

    def update(self, steer_output: dict, drive_output: dict, condlane_output: dict = None, dt: float = 0.033) -> dict:
        model_pts = steer_output.get("model_pts", [])
        
        world_pts = []
        for u, v in model_pts:
            x_fwd, y_lat = self.homography.img_to_world(u, v, is_model_space=True)
            # Filter reasonable road boundaries (0 to 120m fwd, -15m to 15m lat)
            if 1.0 <= x_fwd <= 120.0 and -15.0 <= y_lat <= 15.0:
                world_pts.append((x_fwd, y_lat))

        # Check if CondLane detected ego lane boundaries
        condlane_cte = None
        has_condlane = False
        ldw_state = "NO_WARNING"
        lane_width_m = 3.5

        if condlane_output and condlane_output.get("valid", False):
            metrics = condlane_output.get("metrics", {})
            has_condlane = metrics.get("has_ego_lane", False)
            ldw_state = metrics.get("ldw_state", "NO_WARNING")
            slots = condlane_output.get("slots", {})
            
            # If both ego-left and ego-right detected, project lane center to ground
            if 0 in slots and 1 in slots:
                left_pts = slots[0]["points"]
                right_pts = slots[1]["points"]
                if len(left_pts) > 4 and len(right_pts) > 4:
                    # Ground projection of near points
                    u_l_near, v_l_near = left_pts[-1]
                    u_r_near, v_r_near = right_pts[-1]
                    x_fl, y_ll = self.homography.img_to_world(u_l_near, v_l_near, is_model_space=False)
                    x_fr, y_lr = self.homography.img_to_world(u_r_near, v_r_near, is_model_space=False)
                    
                    if 1.0 <= x_fl <= 40.0 and 1.0 <= x_fr <= 40.0:
                        lane_width_m = abs(y_ll - y_lr)
                        center_y = (y_ll + y_lr) * 0.5
                        condlane_cte = float(center_y)

        path_valid = len(world_pts) >= 6
        raw_cte = 0.0
        raw_yaw = 0.0
        raw_path_curv = 0.0
        poly_coeffs = [0.0, 0.0, 0.0]

        if path_valid:
            pts_arr = np.array(world_pts)
            x_arr = pts_arr[:, 0]
            y_arr = pts_arr[:, 1]

            # Fit 2nd-order polynomial: y = a * x^2 + b * x + c
            try:
                # Weighted fit emphasizing near points
                weights = 1.0 / np.sqrt(np.clip(x_arr, 1.0, 100.0))
                poly_coeffs = np.polyfit(x_arr, y_arr, deg=2, w=weights)
                a, b, c = poly_coeffs[0], poly_coeffs[1], poly_coeffs[2]

                raw_cte = float(c)            # Cross-track error at ego origin x=0
                raw_yaw = float(np.arctan(b)) # Heading angle at origin
                
                # Curvature at lookahead point (e.g. 15m)
                x_eval = 15.0
                dy_dx = 2.0 * a * x_eval + b
                d2y_dx2 = 2.0 * a
                raw_path_curv = float(d2y_dx2 / np.power(1.0 + dy_dx**2, 1.5))
            except Exception:
                path_valid = False

        # Fuse CTE: if CondLaneNet detected real lane markers, blend CTE
        if condlane_cte is not None and path_valid:
            fused_raw_cte = 0.65 * raw_cte + 0.35 * condlane_cte
        elif condlane_cte is not None:
            fused_raw_cte = condlane_cte
        else:
            fused_raw_cte = raw_cte

        # Fuse curvature with AutoDrive if available
        ad_curv = drive_output.get("curvature", 0.0) if drive_output.get("valid", False) else 0.0
        if path_valid and drive_output.get("valid", False):
            meas_curv = 0.70 * raw_path_curv + 0.30 * ad_curv
        elif path_valid:
            meas_curv = raw_path_curv
        else:
            meas_curv = ad_curv

        # Temporal filter update
        if self.prev_time is None:
            self.cte_m = fused_raw_cte
            self.yaw_rad = raw_yaw
            self.fused_curvature = meas_curv
            self.prev_time = 0.0
        else:
            new_cte = (1.0 - self.alpha_state) * self.cte_m + self.alpha_state * fused_raw_cte
            new_yaw = (1.0 - self.alpha_state) * self.yaw_rad + self.alpha_state * raw_yaw
            new_curv = (1.0 - self.alpha_curv) * self.fused_curvature + self.alpha_curv * meas_curv

            if dt > 0.001:
                self.cte_rate_mps = (new_cte - self.cte_m) / dt
                self.yaw_rate_rps = (new_yaw - self.yaw_rad) / dt

            self.cte_m = new_cte
            self.yaw_rad = new_yaw
            self.fused_curvature = new_curv

        return {
            "valid": path_valid or drive_output.get("valid", False) or has_condlane,
            "cte_m": float(self.cte_m),
            "raw_cte_m": float(fused_raw_cte),
            "cte_rate_mps": float(self.cte_rate_mps),
            "yaw_rad": float(self.yaw_rad),
            "raw_yaw_rad": float(raw_yaw),
            "yaw_rate_rps": float(self.yaw_rate_rps),
            "curvature": float(self.fused_curvature),
            "poly_coeffs": [float(p) for p in poly_coeffs],
            "world_pts": world_pts,
            "has_condlane": has_condlane,
            "ldw_state": ldw_state,
            "lane_width_m": float(lane_width_m)
        }

    def reset(self):
        self.cte_m = 0.0
        self.cte_rate_mps = 0.0
        self.yaw_rad = 0.0
        self.yaw_rate_rps = 0.0
        self.fused_curvature = 0.0
        self.prev_time = None
