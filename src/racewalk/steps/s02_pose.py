"""S2 pose.

IN     frames/, 01_frames.json
DO     MediaPipe Pose Landmarker over every frame in VIDEO running mode,
       executed in an ISOLATED subprocess (racewalk.pose_worker)
OUT    02_landmarks.parquet, 02_pose_quality.json
VERIFY one row block per frame, detection rate recorded, frames below the
       visible-landmark floor flagged, confidence distribution reported

Why a subprocess: on Windows, MediaPipe's native bindings fail to initialise
when pyarrow is already loaded in the process (both carry protobuf/abseil).
This process has pandas and pyarrow loaded, so MediaPipe runs next door and
hands back a .npz. See racewalk/pose_worker.py.

Both coordinate spaces are stored:
  x, y, z       normalised image space, origin top-left
  wx, wy, wz    MediaPipe world landmarks, metres, hip-centred
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.io_guard import guarded_path
from racewalk.landmarks import NAMES

N_LANDMARKS = 33

WORKER_HELP = """
The isolated MediaPipe worker failed (exit code {code}).

Check it directly, outside the pipeline:
    .\\.venv\\Scripts\\python.exe -m racewalk.pose_worker --selftest

If that also fails, run:  racewalk diagnose
"""


def _run_worker(ctx: Context, frames_dir: Path, fps: float, out_npz: Path) -> tuple[int, Path]:
    model_path = ctx.cfg.paths.models_dir / ctx.cfg.get("paths.pose_model_file")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe pose model missing: {model_path}\n"
            f"Rerun scripts/bootstrap.ps1, or download pose_landmarker_full.task by hand.")
    options = {
        "delegate": ctx.cfg.get("pose.delegate", "CPU"),
        "num_poses": ctx.cfg.get("pose.num_poses", 1),
        "min_pose_detection_confidence": ctx.cfg.get("pose.min_pose_detection_confidence", 0.5),
        "min_pose_presence_confidence": ctx.cfg.get("pose.min_pose_presence_confidence", 0.5),
        "min_pose_tracking_confidence": ctx.cfg.get("pose.min_pose_tracking_confidence", 0.5),
    }
    cmd = [sys.executable, "-m", "racewalk.pose_worker",
           "--frames-dir", str(frames_dir), "--model", str(model_path),
           "--fps", repr(fps), "--out", str(out_npz),
           "--options", json.dumps(options)]
    # Inherit stdout/stderr so progress streams live to the console.
    return subprocess.run(cmd).returncode, model_path


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S2")
    frames_meta = ctx.read_json("01_frames.json")
    frames_dir = Path(frames_meta["frames_dir"])
    analysis_fps = float(frames_meta["analysis_fps"])
    n_expected = len(sorted(frames_dir.glob("f*.jpg")))

    res.check("frames_available", n_expected > 0, f"{n_expected} frames in {frames_dir}")
    if not n_expected:
        return res

    out_npz = guarded_path(ctx.artefact("02_pose_raw.npz"))
    code, model_path = _run_worker(ctx, frames_dir, analysis_fps, out_npz)
    res.check("pose_worker_exit_ok", code == 0,
              "Isolated MediaPipe worker completed." if code == 0 else WORKER_HELP.format(code=code))
    if code != 0 or not out_npz.is_file():
        return res

    raw = np.load(out_npz, allow_pickle=True)
    arr = raw["landmarks"]
    detected = raw["detected"]
    unreadable = [str(x) for x in raw["unreadable"]]
    n = arr.shape[0]

    track_conf = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    low_conf = float(ctx.cfg.get("quality_gates.landmark_low_confidence_flag", 0.3))
    min_visible = int(ctx.cfg.get("quality_gates.min_visible_landmarks", 20))

    frame_idx = np.repeat(np.arange(n, dtype=np.int32), N_LANDMARKS)
    flat = arr.reshape(n * N_LANDMARKS, 8)
    df = pd.DataFrame({
        "frame": frame_idx,
        "t_ms": (frame_idx * (1000.0 / analysis_fps)).astype(np.float32),
        "lm": np.tile(np.arange(N_LANDMARKS, dtype=np.int16), n),
        "x": flat[:, 0], "y": flat[:, 1], "z": flat[:, 2],
        "visibility": flat[:, 3], "presence": flat[:, 4],
        "wx": flat[:, 5], "wy": flat[:, 6], "wz": flat[:, 7],
    })
    out_parquet = guarded_path(ctx.artefact("02_landmarks.parquet"))
    df.to_parquet(out_parquet, index=False)

    vis = arr[:, :, 3]
    n_visible = np.nansum(vis >= track_conf, axis=1)
    n_low = np.nansum((vis < low_conf) & ~np.isnan(vis), axis=1)
    insufficient = np.where(n_visible < min_visible)[0]
    detection_rate = float(detected.mean())
    usable_rate = float((n_visible >= min_visible).mean())

    quality = {
        "model": model_path.name,
        "execution": "isolated subprocess (racewalk.pose_worker)",
        "delegate": ctx.cfg.get("pose.delegate", "CPU"),
        "n_frames": n,
        "analysis_fps": analysis_fps,
        "detection_rate": detection_rate,
        "usable_frame_rate": usable_rate,
        "min_visible_landmarks_required": min_visible,
        "frames_insufficient_landmarks": insufficient.tolist()[:500],
        "n_frames_insufficient_landmarks": int(insufficient.size),
        "unreadable_frames": unreadable,
        "per_frame_visible_count": n_visible.astype(int).tolist(),
        "per_frame_low_confidence_count": n_low.astype(int).tolist(),
        "per_landmark": {
            NAMES[j]: {
                "mean_visibility": float(np.nanmean(vis[:, j])) if np.isfinite(vis[:, j]).any() else None,
                "frames_above_track_threshold": int(np.nansum(vis[:, j] >= track_conf)),
                "frames_low_confidence": int(np.nansum(vis[:, j] < low_conf)),
            } for j in range(N_LANDMARKS)
        },
        "thresholds": {"track_confidence": track_conf, "low_confidence_flag": low_conf,
                       "min_visible_landmarks": min_visible},
    }
    out_quality = ctx.write_json("02_pose_quality.json", quality)

    res.check("frame_count_matches", n == n_expected, f"worker returned {n} of {n_expected} frames")
    res.check("rows_per_frame_correct", len(df) == n * N_LANDMARKS,
              f"{len(df)} rows for {n} frames x {N_LANDMARKS} landmarks")
    res.check("no_unreadable_frames", not unreadable, f"Unreadable: {unreadable[:10]}")
    res.check("pose_detected_in_most_frames", detection_rate >= 0.80,
              f"Athlete detected in {detection_rate:.1%} of frames. "
              f"Below 80% the video is not usable for consistent measurement.")
    res.check("enough_usable_frames", usable_rate >= 0.60,
              f"{usable_rate:.1%} of frames have at least {min_visible} of 33 landmarks visible.")
    res.check("world_landmarks_present", bool(np.isfinite(arr[:, :, 5]).any()),
              "World landmarks available, joint angles will use metric space.", severity=WARN)
    res.check("low_confidence_share_acceptable", float(np.nanmean(n_low)) <= 8,
              f"Mean {float(np.nanmean(n_low)):.1f} landmarks per frame below the "
              f"{low_conf} confidence flag.", severity=WARN)

    res.outputs["landmarks"] = str(out_parquet)
    res.outputs["pose_quality"] = str(out_quality)
    res.outputs["pose_raw"] = str(out_npz)
    res.stats = {"n_frames": n, "detection_rate": round(detection_rate, 4),
                 "usable_frame_rate": round(usable_rate, 4),
                 "mean_visible_landmarks": round(float(np.mean(n_visible)), 2)}
    return res
