"""Build a run directory from synthetic landmarks and push it through S3..S10."""
from __future__ import annotations

import json
from pathlib import Path

from racewalk import io_guard
from racewalk.config import load_config
from racewalk.context import Context

ROOT = Path(__file__).resolve().parents[1]


def write_frames(run_dir: Path, n: int, size=(640, 360)) -> Path:
    """Plain dark frames with a faint grid, so overlays are visible in tests."""
    import cv2
    import numpy as np
    fdir = run_dir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    w, h = size
    base = np.full((h, w, 3), 48, np.uint8)
    for x in range(0, w, 40):
        cv2.line(base, (x, 0), (x, h), (60, 60, 60), 1)
    for y in range(0, h, 40):
        cv2.line(base, (0, y), (w, y), (60, 60, 60), 1)
    cv2.rectangle(base, (0, int(h * 0.94)), (w, h), (70, 70, 78), -1)      # treadmill belt
    for i in range(n):
        cv2.imwrite(str(fdir / f"f{i:06d}.jpg"), base)
    return fdir


def make_run(run_dir: Path, df, fps: float, session: dict | None = None,
             frames: tuple[int, int] | None = (640, 360), outputs_dir: Path | None = None,
             config_patch: dict | None = None) -> Context:
    run_dir.mkdir(parents=True, exist_ok=True)
    roots = [run_dir] + ([outputs_dir] if outputs_dir else [])
    io_guard.configure(roots)
    df.to_parquet(run_dir / "02_landmarks.parquet", index=False)
    n = int(df["frame"].max()) + 1
    w, h = frames or (1920, 1080)
    (run_dir / "01_frames.json").write_text(json.dumps(
        {"frames_dir": str(run_dir / "frames"), "n_frames": n, "analysis_fps": fps,
         "width": w, "height": h}))
    (run_dir / "02_pose_quality.json").write_text(json.dumps(
        {"detection_rate": 1.0, "per_frame_visible_count": [31] * n}))
    sess = {"athlete_name": "Synthetic Walker", "camera_view": "rear", "setting": "treadmill",
            "level": None, "age": None, "gender": None, "session_date": "2026-09-23",
            "treadmill_speed_kmh": None, "body_mass_kg": None, **(session or {})}
    (run_dir / "00_ingest.json").write_text(json.dumps({
        "video_path": "C:/videos/synthetic.mov", "video_sha256": "0" * 64,
        "measured_fps": fps, "session": sess,
        "probe": {"duration_s": n / fps, "width": w, "height": h,
                  "codec": "h264", "source": "synthetic"}}))
    if frames:
        write_frames(run_dir, n, frames)
    cfg = load_config(ROOT)
    cfg.pipeline["llm"]["retry_backoff_base_s"] = 0     # no waiting in tests
    for dotted, value in (config_patch or {}).items():
        node = cfg.pipeline
        *parents, last = dotted.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    if outputs_dir:
        cfg.paths.outputs_dir = outputs_dir
    return Context(cfg=cfg, run_id=run_dir.name, run_dir=run_dir,
                   video_path=Path("synthetic.mov"), session=sess)
