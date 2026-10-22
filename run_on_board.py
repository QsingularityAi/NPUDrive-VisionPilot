#!/usr/bin/env python3
"""
VisionPilot INT8 On-Board Benchmark & Video Inference Runner (4 Models)
Runs natively on TI J784S4 EVM utilizing TIDLExecutionProvider on C7x/MMA NPU.
Models: AutoSteer, AutoSpeed, AutoDrive, CondLaneNet.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure int8visionpilot is in python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(current_dir) not in sys.path: sys.path.insert(0, str(current_dir))
if str(parent_dir) not in sys.path: sys.path.insert(0, str(parent_dir))

from int8visionpilot.config.config import VisionPilotConfig
from int8visionpilot.pipeline.vision_pilot_pipeline import VisionPilotPipeline

def run_video_benchmark(pipeline: VisionPilotPipeline,
                        video_path: Path,
                        output_dir: Path,
                        num_frames: int = 120,
                        start_frame: int = 0,
                        initial_speed_mps: float = 20.0,
                        hm_thr: float = 0.50,
                        nms_thr: int = 4):
    
    video_name = video_path.stem
    print(f"\n================================================================================")
    print(f"  PROCESSING TEST VIDEO: {video_name} ({video_path.name})")
    print(f"  CondLaneNet Config: Heatmap Threshold = {hm_thr:.2f}, NMS Threshold = {nms_thr}")
    print(f"================================================================================")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_to_process = min(num_frames, total_video_frames - start_frame)
    output_video_path = output_dir / f"{video_name}_visionpilot_int8_hud.mp4"
    output_log_path = output_dir / f"{video_name}_telemetry.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (1920, 1080))

    pipeline.reset()
    ego_speed = initial_speed_mps

    latencies_steer = []
    latencies_speed = []
    latencies_drive = []
    latencies_cond = []
    latencies_total = []
    aeb_triggers = 0
    cipo_detections = 0
    lane_detections = 0
    ldw_warnings = 0
    lane_widths = []
    telemetry_records = []

    print(f"Processing {frames_to_process} frames from frame {start_frame} at {fps:.1f} FPS...")

    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = start_frame + i
        hud_frame, telem = pipeline.process_frame(
            frame,
            ego_speed_mps=ego_speed,
            frame_idx=frame_idx,
            video_name=video_name,
            dt=dt,
            hm_thr=hm_thr,
            nms_thr=nms_thr,
            render_hud=True
        )

        writer.write(hud_frame)

        # Closed-loop speed integration: update speed based on planner acceleration
        accel = telem["plan"]["acceleration_mps2"]
        ego_speed = max(0.0, ego_speed + accel * dt)

        # Collect metrics
        steer_lat = telem["steer_res"].get("latency_ms", 0.0)
        speed_lat = telem["speed_res"].get("latency_ms", 0.0)
        drive_lat = telem["drive_res"].get("latency_ms", 0.0)
        cond_lat  = telem["condlane_res"].get("latency_ms", 0.0) if telem.get("condlane_res") else 0.0
        total_lat = telem["total_pipeline_ms"]

        latencies_steer.append(steer_lat)
        latencies_speed.append(speed_lat)
        latencies_drive.append(drive_lat)
        latencies_cond.append(cond_lat)
        latencies_total.append(total_lat)

        if telem["plan"].get("aeb_active", False):
            aeb_triggers += 1
        if telem["cipo_est"].get("has_cipo", False):
            cipo_detections += 1
        if telem.get("condlane_res", {}).get("valid", False):
            lane_detections += 1
            lw = telem["lat_est"].get("lane_width_m", 3.5)
            lane_widths.append(lw)
        if "WARN" in telem["lat_est"].get("ldw_state", ""):
            ldw_warnings += 1

        # Keep lightweight telemetry record
        telemetry_records.append({
            "frame": frame_idx,
            "steer_lat_ms": round(steer_lat, 2),
            "speed_lat_ms": round(speed_lat, 2),
            "drive_lat_ms": round(drive_lat, 2),
            "cond_lat_ms": round(cond_lat, 2),
            "total_lat_ms": round(total_lat, 2),
            "ego_speed_kmh": round(ego_speed * 3.6, 2),
            "steer_deg": round(telem["plan"]["steering_angle_deg"], 2),
            "cte_m": round(telem["lat_est"]["cte_m"], 3),
            "curvature": round(telem["lat_est"]["curvature"], 5),
            "lane_width_m": round(telem["lat_est"].get("lane_width_m", 3.5), 2),
            "ldw_state": telem["lat_est"].get("ldw_state", "NO_WARNING"),
            "cipo_dist_m": round(telem["cipo_est"]["distance_m"], 2),
            "ttc_s": round(telem["plan"]["ttc_s"], 2),
            "aeb_state": telem["plan"]["aeb_state"],
            "accel_mps2": round(accel, 2)
        })

        if (i + 1) % 25 == 0 or (i + 1) == frames_to_process:
            print(f"  Frame {i+1:3d}/{frames_to_process} | "
                  f"NPU: Steer {steer_lat:4.1f}ms, Speed {speed_lat:4.1f}ms, Drive {drive_lat:4.1f}ms, CondLane {cond_lat:4.1f}ms | "
                  f"Total: {total_lat:4.1f}ms ({1000.0/total_lat:4.1f} FPS) | "
                  f"Lanes: {telem.get('condlane_res', {}).get('num_lanes', 0)} | "
                  f"CIPO: {telem['cipo_est']['distance_m']:5.1f}m | "
                  f"AEB: {telem['plan']['aeb_state']}")

    writer.release()
    cap.release()

    # Save JSON log
    with open(output_log_path, "w") as f:
        json.dump(telemetry_records, f, indent=2)

    avg_steer = float(np.mean(latencies_steer))
    avg_speed = float(np.mean(latencies_speed))
    avg_drive = float(np.mean(latencies_drive))
    avg_cond  = float(np.mean(latencies_cond))
    avg_total = float(np.mean(latencies_total))
    fps_pipeline = 1000.0 / avg_total if avg_total > 0 else 0.0
    avg_lw = float(np.mean(lane_widths)) if lane_widths else 3.5

    print(f"\n[+] FINISHED: {video_name}")
    print(f"    Rendered Video : {output_video_path} ({os.path.getsize(output_video_path) / 1024 / 1024:.2f} MB)")
    print(f"    Telemetry JSON : {output_log_path}")
    print(f"    Avg Latencies  : Steer={avg_steer:.1f}ms, Speed={avg_speed:.1f}ms, Drive={avg_drive:.1f}ms, CondLane={avg_cond:.1f}ms")
    print(f"    Pipeline Total : {avg_total:.1f} ms  ({fps_pipeline:.1f} FPS)")
    print(f"    Lanes Detected : {lane_detections}/{frames_to_process} frames (Avg Width: {avg_lw:.2f}m)")
    print(f"    CIPO Detected  : {cipo_detections}/{frames_to_process} frames")
    print(f"    AEB Active     : {aeb_triggers}/{frames_to_process} frames")
    print(f"    LDW Warnings   : {ldw_warnings}/{frames_to_process} frames")

    return {
        "video": video_name,
        "frames_processed": frames_to_process,
        "hm_thr": hm_thr,
        "nms_thr": nms_thr,
        "avg_steer_ms": avg_steer,
        "avg_speed_ms": avg_speed,
        "avg_drive_ms": avg_drive,
        "avg_cond_ms": avg_cond,
        "avg_total_ms": avg_total,
        "fps": fps_pipeline,
        "lane_frames": lane_detections,
        "avg_lane_width_m": avg_lw,
        "cipo_frames": cipo_detections,
        "aeb_frames": aeb_triggers,
        "ldw_frames": ldw_warnings,
        "video_file": output_video_path.name
    }

def main():
    parser = argparse.ArgumentParser(description="VisionPilot Quad INT8 NPU Pipeline Benchmark (4 Models)")
    parser.add_argument("--frames", type=int, default=300, help="Frames per video to process")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame number")
    parser.add_argument("--full", action="store_true", help="Process full video")
    parser.add_argument("--hm-thr", type=float, default=0.50, help="CondLane heatmap threshold")
    parser.add_argument("--nms-thr", type=int, default=4, help="CondLane NMS seed grouping threshold")
    parser.add_argument("--sweep-thresholds", action="store_true", help="Perform threshold sensitivity sweep")
    parser.add_argument("--output-dir", type=str, default="/home/root/test_npu/int8visionpilot/outputs")
    parser.add_argument("--video-dir", type=str, default="/home/root/test_npu/int8visionpilot/test video")
    parser.add_argument("--video-name", type=str, default="AEB_Moving.mp4", help="Specific video filename or 'all'")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vid_dir = Path(args.video_dir)
    if args.video_name == "all":
        candidate_videos = [
            vid_dir / "AEB_Moving.mp4",
            vid_dir / "AEB_Stationary.mp4",
            vid_dir / "13588981_3840_2160_30fps-2.mp4",
            vid_dir / "15050854_3840_2160_30fps-2.mp4"
        ]
    else:
        candidate_videos = [vid_dir / args.video_name]
    
    videos = []
    for cv_p in candidate_videos:
        if cv_p.exists():
            videos.append(cv_p)
        else:
            alt = Path("/home/root/test_npu/int8visionpilot/test video") / cv_p.name
            if alt.exists():
                videos.append(alt)
            else:
                alt2 = Path("/home/root/simulationvideo") / cv_p.name
                if alt2.exists(): videos.append(alt2)

    print(f"[+] Found {len(videos)} test videos: {[v.name for v in videos]}")

    # Initialize VisionPilot Quad INT8 Pipeline
    pipeline = VisionPilotPipeline(VisionPilotConfig.BOARD_MODELS, prefer_tidl=True)

    results = []

    if args.sweep_thresholds:
        print("\n================================================================================")
        print("          RUNNING CONDLANENET THRESHOLD SENSITIVITY BENCHMARK                   ")
        print("================================================================================")
        # Threshold grid: hm_thr in [0.35, 0.50, 0.65] with nms_thr=4
        test_vid = videos[0]
        threshold_results = []
        for h_thr in [0.35, 0.50, 0.65]:
            for n_thr in [3, 4, 5]:
                print(f"\n--- Testing hm_thr={h_thr:.2f}, nms_thr={n_thr} on {test_vid.name} (40 frames) ---")
                res = run_video_benchmark(pipeline, test_vid, out_dir, num_frames=40, start_frame=180,
                                          hm_thr=h_thr, nms_thr=n_thr)
                threshold_results.append(res)
        
        sweep_path = out_dir / "threshold_sweep_summary.json"
        with open(sweep_path, "w") as f:
            json.dump(threshold_results, f, indent=2)
        print(f"\n[✓] Threshold sensitivity sweep saved to {sweep_path}")

    # Run full/standard benchmark on both test videos
    for v_path in videos:
        num_f = 99999 if args.full else args.frames
        start_f = args.start_frame if not args.full else 0
        res = run_video_benchmark(pipeline, v_path, out_dir, num_frames=num_f, start_frame=start_f,
                                  hm_thr=args.hm_thr, nms_thr=args.nms_thr)
        results.append(res)

    # Save summary report
    summary_path = out_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=========================================================================================")
    print("                    VISIONPILOT QUAD INT8 HARDWARE BENCHMARK SUMMARY                     ")
    print("=========================================================================================")
    print(f"{'Video':<16} | {'Steer':<7} | {'Speed':<7} | {'Drive':<7} | {'CondLane':<8} | {'Total':<7} | {'FPS':<5} | {'Lanes':<8} | {'CIPO':<8}")
    print("-" * 89)
    for r in results:
        print(f"{r['video']:<16} | {r['avg_steer_ms']:<5.1f}ms | {r['avg_speed_ms']:<5.1f}ms | {r['avg_drive_ms']:<5.1f}ms | {r['avg_cond_ms']:<6.1f}ms | {r['avg_total_ms']:<5.1f}ms | {r['fps']:<5.1f} | {r['lane_frames']}/{r['frames_processed']:<4} | {r['cipo_frames']}/{r['frames_processed']:<4}")
    print("=========================================================================================")
    print(f"[✓] All benchmarks completed successfully. Outputs saved in: {out_dir}\n")

if __name__ == "__main__":
    main()
