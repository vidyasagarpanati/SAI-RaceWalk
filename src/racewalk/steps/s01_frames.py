"""S1 frames.

IN     00_ingest.json
DO     one ffmpeg decode pass to JPEG at the configured analysis rate
OUT    frames/f######.jpg, 01_frames.json
VERIFY frame count within tolerance of expectation, no zero-byte frames, first
       and last frame decodable, source untouched (hash re-checked)

The video is decoded exactly once for the whole pipeline. S2, S6 and S7 all read
these JPEGs, so a rerun of any later step never re-decodes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from racewalk.config import sha256_file
from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.io_guard import guarded_path
from racewalk.steps.s00_ingest import _ffmpeg_exe

FRAME_GLOB = "f*.jpg"


def _ffmpeg_quality(jpeg_quality: int) -> int:
    """Map a 0-100 quality to ffmpeg's mjpeg -q:v scale (2 best, 31 worst)."""
    jpeg_quality = max(1, min(100, int(jpeg_quality)))
    return max(2, min(31, round(2 + (100 - jpeg_quality) * (29 / 99))))


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S1")
    ingest = ctx.read_json("00_ingest.json")
    video = Path(ingest["video_path"])

    native_fps = float(ingest["measured_fps"] or 0)
    target_fps = float(ctx.cfg.get("frames.analysis_fps", 60))
    analysis_fps = min(native_fps, target_fps) if native_fps else target_fps
    duration = ingest["probe"].get("duration_s")
    max_frames = int(ctx.cfg.get("frames.max_frames", 20000))

    frames_dir = guarded_path(ctx.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frames_dir.glob(FRAME_GLOB))
    if not existing:
        cmd = [
            _ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"fps={analysis_fps}",
            "-q:v", str(_ffmpeg_quality(ctx.cfg.get("frames.jpeg_quality", 92))),
            "-frames:v", str(max_frames),
            "-start_number", "0",
            str(frames_dir / "f%06d.jpg"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        res.check("ffmpeg_decode_ok", proc.returncode == 0,
                  (proc.stderr or "").strip()[:2000] or "ffmpeg decode completed.")
        if proc.returncode != 0:
            return res
        existing = sorted(frames_dir.glob(FRAME_GLOB))
    else:
        res.check("ffmpeg_decode_ok", True,
                  f"Reused {len(existing)} frames already decoded for this run.")

    n = len(existing)
    res.check("frames_produced", n > 0, f"{n} frames at {analysis_fps:g} fps")
    if n == 0:
        return res

    if duration:
        expected = min(int(round(duration * analysis_fps)), max_frames)
        tolerance = max(2, int(0.02 * expected))
        res.check("frame_count_matches_duration", abs(n - expected) <= tolerance,
                  f"Got {n}, expected about {expected} (tolerance {tolerance}).",
                  severity=WARN, evidence={"got": n, "expected": expected})

    empty = [p.name for p in existing if p.stat().st_size == 0]
    res.check("no_zero_byte_frames", not empty, f"Zero-byte frames: {empty[:10]}")

    import cv2
    first_img = cv2.imread(str(existing[0]))
    ok_first = first_img is not None
    ok_last = cv2.imread(str(existing[-1])) is not None
    height_px, width_px = (first_img.shape[:2] if ok_first else (None, None))
    res.check("endpoints_decodable", ok_first and ok_last,
              f"first={ok_first} last={ok_last}")

    res.check("source_unmodified", sha256_file(video) == ingest["video_sha256"],
              "Source video hash unchanged after decoding.")

    res.check("frame_cap_not_hit", n < max_frames,
              f"Hit the {max_frames}-frame cap. The video is longer than the "
              f"configured analysis window and is truncated.",
              severity=WARN)

    payload = {
        "frames_dir": str(frames_dir),
        "n_frames": n,
        "analysis_fps": analysis_fps,
        "native_fps": native_fps,
        "width": width_px,
        "height": height_px,
        "frame_pattern": "f%06d.jpg",
        "first_frame": existing[0].name,
        "last_frame": existing[-1].name,
        "timestamps_s": None,
        "note": "Frame i corresponds to t = i / analysis_fps seconds in the source.",
    }
    out = ctx.write_json("01_frames.json", payload)
    res.outputs["frames_index"] = str(out)
    res.outputs["frames_dir"] = str(frames_dir)
    res.stats = {"n_frames": n, "analysis_fps": analysis_fps}
    return res
