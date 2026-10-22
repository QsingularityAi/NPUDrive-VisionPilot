"""Vehicle Control Planning: IDM Adaptive Cruise Control & AEB Safety Guard."""
import numpy as np
from ..config.config import VisionPilotConfig

class VisionPilotPlanner:
    def __init__(self, speed_limit_mps: float = None, wheelbase_m: float = None):
        self.v_limit = speed_limit_mps if speed_limit_mps else VisionPilotConfig.SPEED_LIMIT_MPS
        self.L = wheelbase_m if wheelbase_m else VisionPilotConfig.WHEELBASE_M
        
        # IDM parameters
        self.s0 = VisionPilotConfig.IDM_S0
        self.T = VisionPilotConfig.IDM_T
        self.a_max = VisionPilotConfig.IDM_A_MAX
        self.b_comf = VisionPilotConfig.IDM_B_COMFORT
        self.delta = VisionPilotConfig.IDM_DELTA
        
        # Stanley parameters
        self.k_cte = 0.45
        self.k_us = 0.0015  # Understeer gradient rad/(m/s^2)
        self.max_steer_rad = 0.55  # ~31.5 degrees

    def compute_plan(self, lateral_est: dict, cipo_est: dict, ego_v_mps: float) -> dict:
        cte = lateral_est.get("cte_m", 0.0)
        yaw = lateral_est.get("yaw_rad", 0.0)
        kappa = lateral_est.get("curvature", 0.0)
        
        has_cipo = cipo_est.get("has_cipo", False)
        cipo_dist = cipo_est.get("distance_m", VisionPilotConfig.AUTODRIVE_D_MAX_M)
        cipo_v_rel = cipo_est.get("velocity_ms", 0.0)
        ttc_s = cipo_est.get("ttc_s", 99.9)

        # ─── 1. Lateral Control (Stanley Steering) ───────────────────────────
        # delta = yaw_error + arctan(k * cte / (v + 1)) + feedforward curvature
        v_eff = max(ego_v_mps, 0.5)
        steer_cte_term = float(np.arctan2(self.k_cte * cte, v_eff + 1.0))
        steer_ff_term = float((self.L * kappa) / (1.0 + self.k_us * (v_eff ** 2)))
        steering_angle_rad = float(np.clip(yaw + steer_cte_term + steer_ff_term,
                                          -self.max_steer_rad, self.max_steer_rad))
        steering_angle_deg = float(np.degrees(steering_angle_rad))

        # ─── 2. Longitudinal Control (IDM ACC) ──────────────────────────────
        # Road curvature speed governor
        abs_kappa = max(abs(kappa), 1e-5)
        v_curv_max = float(np.sqrt(VisionPilotConfig.MU * VisionPilotConfig.GRAVITY / abs_kappa))
        v_target = min(self.v_limit, v_curv_max)

        # Closing speed: positive when ego is faster than lead vehicle
        delta_v = -cipo_v_rel

        # Desired dynamic gap
        dynamic_term = (v_eff * delta_v) / (2.0 * np.sqrt(self.a_max * self.b_comf))
        s_star = self.s0 + max(0.0, v_eff * self.T + dynamic_term)
        s = max(0.5, cipo_dist)

        # IDM terms
        free_road_term = (v_eff / max(v_target, 1.0)) ** self.delta
        interaction_term = ((s_star / s) ** 2.0) if has_cipo else 0.0

        idm_accel = self.a_max * (1.0 - free_road_term - interaction_term)

        # ─── 3. Autonomous Emergency Braking (AEB) Safety Guardian ──────────
        aeb_state = "NORMAL"
        aeb_active = False
        target_accel = idm_accel

        if has_cipo:
            if ttc_s < VisionPilotConfig.AEB_STAGE2_TTC_S or (cipo_dist < 8.0 and delta_v > 1.0):
                # Critical imminent collision -> Maximum Emergency Braking
                aeb_state = "AEB_STAGE2_EMERGENCY"
                target_accel = VisionPilotConfig.AEB_EMERGENCY_ACCEL
                aeb_active = True
            elif ttc_s < VisionPilotConfig.AEB_STAGE1_TTC_S or (cipo_dist < 15.0 and delta_v > 2.0):
                # High risk -> Automatic brake intervention
                aeb_state = "AEB_STAGE1_INTERVENTION"
                target_accel = min(idm_accel, -4.5)
                aeb_active = True
            elif ttc_s < VisionPilotConfig.AEB_FCW_TTC_S:
                # Forward collision warning
                aeb_state = "AEB_FCW_WARNING"
                target_accel = min(idm_accel, -2.0)
                aeb_active = False

        # Limit acceleration bounds [-8.0 m/s^2, +2.5 m/s^2]
        target_accel = float(np.clip(target_accel, -8.0, 2.5))

        return {
            "steering_angle_rad": steering_angle_rad,
            "steering_angle_deg": steering_angle_deg,
            "acceleration_mps2": target_accel,
            "idm_accel_mps2": float(idm_accel),
            "target_velocity_mps": float(v_target),
            "aeb_state": aeb_state,
            "aeb_active": aeb_active,
            "ttc_s": float(ttc_s),
            "cipo_distance_m": float(cipo_dist) if has_cipo else 999.0
        }
