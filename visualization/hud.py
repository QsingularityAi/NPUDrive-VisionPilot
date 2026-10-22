"""High-End Automotive ADAS HUD & Augmented Reality Visualizer."""
import cv2
import numpy as np
from ..config.config import VisionPilotConfig
from ..calibration.homography import top_crop_2_1

class VisionPilotHUD:
    def __init__(self, out_width: int = 1920, out_height: int = 1080):
        self.W = out_width
        self.H = out_height

    def render_frame(self, raw_frame_bgr: np.ndarray,
                     drive_res: dict, steer_res: dict, speed_res: dict,
                     lateral_est: dict, cipo_est: dict, plan: dict,
                     ego_speed_mps: float, frame_idx: int,
                     condlane_res: dict = None,
                     video_name: str = "Video") -> np.ndarray:
        
        canvas = cv2.resize(raw_frame_bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        raw_h, raw_w = raw_frame_bgr.shape[:2]
        crop_top = top_crop_2_1(raw_h, raw_w)
        crop_h = float(raw_h - crop_top)

        # Coordinate mapper from model space (1024x512) to canvas space (1920x1080)
        def model_to_canvas(u_m, v_m):
            u_raw = u_m * (float(raw_w) / float(VisionPilotConfig.NET_W))
            v_raw = float(crop_top) + v_m * (crop_h / float(VisionPilotConfig.NET_H))
            x_c = int(round(u_raw * (float(self.W) / float(raw_w))))
            y_c = int(round(v_raw * (float(self.H) / float(raw_h))))
            return x_c, y_c

        # Frame coordinate mapper to canvas space (1920x1080)
        def frame_to_canvas(u_f, v_f):
            x_c = int(round(u_f * (float(self.W) / float(raw_w))))
            y_c = int(round(v_f * (float(self.H) / float(raw_h))))
            return x_c, y_c

        # ─── 1. AR Road Trajectory Overlay (AutoSteer Waypoints & Poly) ─────
        pts_canvas = []
        for u_m, v_m in steer_res.get("model_pts", []):
            xc, yc = model_to_canvas(u_m, v_m)
            if 0 <= xc < self.W and 0 <= yc < self.H:
                pts_canvas.append((xc, yc))

        if len(pts_canvas) > 2:
            pts_arr = np.array(pts_canvas, dtype=np.int32)
            ribbon_overlay = canvas.copy()
            ribbon_color = (0, 230, 255) if not plan.get("aeb_active", False) else (0, 80, 255)
            
            # Thick road centerline
            cv2.polylines(ribbon_overlay, [pts_arr], isClosed=False, color=ribbon_color, thickness=6, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pts_arr], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            
            # Draw glow circles at waypoints
            for pt in pts_canvas[::3]:
                cv2.circle(canvas, pt, 4, ribbon_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, pt, 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.addWeighted(ribbon_overlay, 0.35, canvas, 0.65, 0, canvas)

        # ─── 2. CondLaneNet Lane Boundaries (Ego Left, Right & Outer) ────────
        ldw_state = lateral_est.get("ldw_state", "NO_WARNING")
        if condlane_res and condlane_res.get("valid", False):
            slots = condlane_res.get("slots", {})
            lane_overlay = canvas.copy()
            
            for slot_id, slot_data in slots.items():
                pts_f = slot_data["points"]
                pts_c = [frame_to_canvas(p[0], p[1]) for p in pts_f]
                valid_pts = np.array([p for p in pts_c if 0 <= p[0] < self.W and 0 <= p[1] < self.H], dtype=np.int32)
                
                if len(valid_pts) > 2:
                    if slot_id == 0:  # Ego-Left
                        l_col = (0, 140, 255) if ldw_state == "WARN_LEFT" else (80, 240, 100)
                        cv2.polylines(lane_overlay, [valid_pts], isClosed=False, color=l_col, thickness=5, lineType=cv2.LINE_AA)
                        cv2.polylines(canvas, [valid_pts], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    elif slot_id == 1:  # Ego-Right
                        r_col = (0, 140, 255) if ldw_state == "WARN_RIGHT" else (80, 240, 100)
                        cv2.polylines(lane_overlay, [valid_pts], isClosed=False, color=r_col, thickness=5, lineType=cv2.LINE_AA)
                        cv2.polylines(canvas, [valid_pts], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    else:  # Outer lanes
                        cv2.polylines(lane_overlay, [valid_pts], isClosed=False, color=(200, 200, 80), thickness=3, lineType=cv2.LINE_AA)

            cv2.addWeighted(lane_overlay, 0.40, canvas, 0.60, 0, canvas)

        # ─── 3. Detected Objects & CIPO Reticles (AutoSpeed Detections) ──────
        detections = speed_res.get("detections", [])
        best_cipo = cipo_est.get("best_cipo")
        
        for det in detections:
            x1_m, y1_m = det["x1"], det["y1"]
            x2_m, y2_m = det["x2"], det["y2"]
            
            c_x1, c_y1 = model_to_canvas(x1_m, y1_m)
            c_x2, c_y2 = model_to_canvas(x2_m, y2_m)
            
            is_cipo = (best_cipo is not None and 
                       abs(det["x1"] - best_cipo["x1"]) < 5 and 
                       abs(det["y1"] - best_cipo["y1"]) < 5)
            
            box_col = (0, 0, 255) if (is_cipo and plan.get("aeb_active", False)) else ((0, 215, 255) if is_cipo else (180, 180, 50))
            box_thickness = 3 if is_cipo else 2
            
            cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), box_col, box_thickness, lineType=cv2.LINE_AA)
            
            cipo_dist = det.get("x_fwd", 0.0)
            dist_str = f" | {cipo_dist:.1f}m" if cipo_dist > 0.1 else ""
            badge_text = f"{det['class_name']}{dist_str}"
            
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(canvas, (c_x1, max(0, c_y1 - 24)), (c_x1 + tw + 10, c_y1), box_col, -1)
            cv2.putText(canvas, badge_text, (c_x1 + 5, c_y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            
            if is_cipo:
                bc_x, bc_y = model_to_canvas(det["bottom_center"][0], det["bottom_center"][1])
                cv2.circle(canvas, (bc_x, bc_y), 8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (bc_x, bc_y), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        # ─── 4. Top Glassmorphic Navigation Banner ───────────────────────────
        banner_h = 75
        banner_overlay = canvas[:banner_h, :].copy()
        cv2.rectangle(banner_overlay, (0, 0), (self.W, banner_h), (18, 22, 28), -1)
        cv2.addWeighted(banner_overlay, 0.88, canvas[:banner_h, :], 0.12, 0, canvas[:banner_h, :])
        
        cv2.line(canvas, (0, banner_h), (self.W, banner_h), (60, 75, 95), 1, cv2.LINE_AA)
        
        cv2.putText(canvas, "VISIONPILOT INT8 ADAS", (30, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "QUAD INT8 TIDL PIPELINE (4 MODELS)  *  TI J784S4 C7x/MMA NPU", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 160), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Source: {video_name} | Frame #{frame_idx:04d}", (self.W // 2 - 140, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 215, 230), 1, cv2.LINE_AA)

        # Hardware Badge (Top Right)
        badge_w, badge_h = 280, 48
        bx1, by1 = self.W - badge_w - 25, 14
        cv2.rectangle(canvas, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (30, 38, 48), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (60, 210, 120), 1)
        cv2.putText(canvas, "TI J784S4 NPU [4x INT8]", (bx1 + 18, by1 + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (80, 240, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "TIDLExecutionProvider", (bx1 + 18, by1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 195, 210), 1, cv2.LINE_AA)

        # ─── 5. Left Instrument & Telemetry Panel (Glassmorphism) ────────────
        panel_w, panel_h = 320, 290
        px1, py1 = 30, banner_h + 20
        panel_overlay = canvas[py1:py1+panel_h, px1:px1+panel_w].copy()
        cv2.rectangle(panel_overlay, (0, 0), (panel_w, panel_h), (16, 20, 26), -1)
        cv2.addWeighted(panel_overlay, 0.82, canvas[py1:py1+panel_h, px1:px1+panel_w], 0.18, 0, canvas[py1:py1+panel_h, px1:px1+panel_w])
        cv2.rectangle(canvas, (px1, py1), (px1 + panel_w, py1 + panel_h), (50, 65, 80), 1)

        cv2.putText(canvas, "VEHICLE TELEMETRY", (px1 + 16, py1 + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (px1 + 16, py1 + 32), (px1 + panel_w - 16, py1 + 32), (50, 65, 80), 1)

        ego_kmh = ego_speed_mps * 3.6
        steer_deg = plan.get("steering_angle_deg", 0.0)
        cte_cm = lateral_est.get("cte_m", 0.0) * 100.0
        curv = lateral_est.get("curvature", 0.0)
        accel = plan.get("acceleration_mps2", 0.0)
        lane_w_m = lateral_est.get("lane_width_m", 3.5)

        lines_left = [
            ("Ego Speed", f"{ego_kmh:.1f} km/h", (255, 255, 255)),
            ("Steering Angle", f"{steer_deg:+.1f} deg", (0, 225, 255) if abs(steer_deg) > 5.0 else (220, 220, 220)),
            ("Cross-Track Err", f"{cte_cm:+.1f} cm", (240, 200, 80)),
            ("Road Curvature", f"{curv:+.4f} 1/m", (200, 220, 240)),
            ("Lane Width", f"{lane_w_m:.2f} m", (100, 240, 160) if lateral_est.get("has_condlane", False) else (160, 160, 160)),
            ("LDW Status", ldw_state.replace("NO_WARNING", "NOMINAL"), (0, 140, 255) if "WARN" in ldw_state else (80, 240, 100)),
            ("Target Accel", f"{accel:+.2f} m/s2", (0, 80, 255) if accel < -2.0 else ((80, 240, 100) if accel > 0 else (200, 200, 200))),
        ]

        for idx, (label, val, val_col) in enumerate(lines_left):
            y_pos = py1 + 56 + idx * 33
            cv2.putText(canvas, label, (px1 + 16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 175, 190), 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (px1 + panel_w - 130, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.50, val_col, 1, cv2.LINE_AA)

        # ─── 6. Right Hardware NPU Benchmark Panel (4 Models) ────────────────
        r_panel_w, r_panel_h = 340, 290
        r_px1 = self.W - r_panel_w - 30
        r_py1 = banner_h + 20
        r_overlay = canvas[r_py1:r_py1+r_panel_h, r_px1:r_px1+r_panel_w].copy()
        cv2.rectangle(r_overlay, (0, 0), (r_panel_w, r_panel_h), (16, 20, 26), -1)
        cv2.addWeighted(r_overlay, 0.82, canvas[r_py1:r_py1+r_panel_h, r_px1:r_px1+r_panel_w], 0.18, 0, canvas[r_py1:r_py1+r_panel_h, r_px1:r_px1+r_panel_w])
        cv2.rectangle(canvas, (r_px1, r_py1), (r_px1 + r_panel_w, r_py1 + r_panel_h), (50, 65, 80), 1)

        cv2.putText(canvas, "TI J784S4 NPU BENCHMARK", (r_px1 + 16, r_py1 + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (80, 240, 150), 1, cv2.LINE_AA)
        cv2.line(canvas, (r_px1 + 16, r_py1 + 32), (r_px1 + r_panel_w - 16, r_py1 + 32), (50, 65, 80), 1)

        lat_steer = steer_res.get("latency_ms", 0.0)
        lat_speed = speed_res.get("latency_ms", 0.0)
        lat_drive = drive_res.get("latency_ms", 0.0)
        lat_cond  = condlane_res.get("latency_ms", 0.0) if condlane_res else 0.0
        total_lat = lat_steer + lat_speed + lat_drive + lat_cond
        live_fps = 1000.0 / total_lat if total_lat > 0.1 else 0.0

        lines_right = [
            ("AutoSteer NPU", f"{lat_steer:.1f} ms", (240, 240, 240)),
            ("AutoSpeed NPU", f"{lat_speed:.1f} ms", (240, 240, 240)),
            ("AutoDrive NPU", f"{lat_drive:.1f} ms", (240, 240, 240)),
            ("CondLane NPU",  f"{lat_cond:.1f} ms",  (240, 240, 240)),
            ("Pipeline Total", f"{total_lat:.1f} ms", (0, 220, 255)),
            ("Throughput", f"{live_fps:.1f} FPS", (80, 240, 150)),
        ]

        for idx, (label, val, val_col) in enumerate(lines_right):
            y_pos = r_py1 + 56 + idx * 37
            cv2.putText(canvas, label, (r_px1 + 16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 175, 190), 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (r_px1 + r_panel_w - 110, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.52, val_col, 1, cv2.LINE_AA)

        # ─── 6. High-Visibility Hazard / AEB Banner ──────────────────────────
        aeb_state = plan.get("aeb_state", "NORMAL")
        ttc_s = plan.get("ttc_s", 99.9)
        cipo_d = cipo_est.get("distance_m", 999.0)

        if aeb_state == "AEB_STAGE2_EMERGENCY":
            alert_w, alert_h = 760, 64
            ax1 = (self.W - alert_w) // 2
            ay1 = banner_h + 30
            
            # Pulsing bright red emergency alert
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), (0, 0, 230), -1)
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), (255, 255, 255), 2)
            alert_msg = f"[!] EMERGENCY BRAKING ACTIVE: TTC {ttc_s:.1f}s | GAP {cipo_d:.1f}m"
            cv2.putText(canvas, alert_msg, (ax1 + 25, ay1 + 42),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        elif aeb_state == "AEB_STAGE1_INTERVENTION":
            alert_w, alert_h = 700, 56
            ax1 = (self.W - alert_w) // 2
            ay1 = banner_h + 30
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), (0, 120, 255), -1)
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), (255, 255, 255), 1)
            alert_msg = f"[!] BRAKE INTERVENTION: TTC {ttc_s:.1f}s | GAP {cipo_d:.1f}m"
            cv2.putText(canvas, alert_msg, (ax1 + 25, ay1 + 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        elif aeb_state == "AEB_FCW_WARNING":
            alert_w, alert_h = 660, 50
            ax1 = (self.W - alert_w) // 2
            ay1 = banner_h + 30
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), (0, 190, 240), -1)
            alert_msg = f"COLLISION WARNING: TTC {ttc_s:.1f}s | GAP {cipo_d:.1f}m"
            cv2.putText(canvas, alert_msg, (ax1 + 25, ay1 + 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)

        # ─── 7. Bottom ADAS Status Strip ────────────────────────────────────
        strip_h = 55
        sy1 = self.H - strip_h
        strip_overlay = canvas[sy1:self.H, :].copy()
        cv2.rectangle(strip_overlay, (0, 0), (self.W, strip_h), (12, 16, 22), -1)
        cv2.addWeighted(strip_overlay, 0.90, canvas[sy1:self.H, :], 0.10, 0, canvas[sy1:self.H, :])
        cv2.line(canvas, (0, sy1), (self.W, sy1), (45, 55, 70), 1, cv2.LINE_AA)

        cipo_v_kmh = cipo_est.get("velocity_ms", 0.0) * 3.6
        cipo_text = f"CIPO DIST: {cipo_d:.1f} m  |  REL VEL: {cipo_v_kmh:+.1f} km/h" if cipo_est.get("has_cipo", False) else "CIPO: NO IN-PATH OBSTACLE"
        ttc_text = f"TTC: {ttc_s:.1f} s" if ttc_s < 30.0 else "TTC: CLEAR"
        mode_text = f"SAFETY STATE: {aeb_state}"
        
        cv2.putText(canvas, cipo_text, (40, sy1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, ttc_text, (self.W // 2 - 80, sy1 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.60,
                    (0, 80, 255) if ttc_s < 2.5 else (80, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, mode_text, (self.W - 360, sy1 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.58,
                    (0, 0, 255) if "EMERGENCY" in aeb_state else ((0, 160, 255) if "INTERVENTION" in aeb_state else (80, 240, 140)), 1, cv2.LINE_AA)

        return canvas
