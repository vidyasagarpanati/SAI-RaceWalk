"""S0 ingest.

IN     source video path, session.json
DO     hash the video, probe its real properties, validate session metadata
OUT    00_ingest.json
VERIFY file readable, hash recorded, measured fps plausible, required session
       fields present or explicitly marked NOT PROVIDED, source directory is not
       on the writable whitelist
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from racewalk import io_guard
from racewalk.config import sha256_file
from racewalk.context import Context
from racewalk.contracts import WARN, StepResult

NOT_PROVIDED = "NOT PROVIDED - CANNOT BE CONFIRMED"

REQUIRED_SESSION_FIELDS = [
    "athlete_name", "age", "gender", "level", "setting", "camera_view", "session_date",
]


def _ffprobe_exe() -> str | None:
    return shutil.which("ffprobe")


def _ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "No ffmpeg found. Install it (winget install Gyan.FFmpeg) or ensure "
            "imageio-ffmpeg is installed."
        ) from exc


def _probe_ffprobe(video: Path) -> dict[str, Any] | None:
    exe = _ffprobe_exe()
    if not exe:
        return None
    cmd = [exe, "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(video)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    except Exception:
        return None
    data = json.loads(out.stdout)
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        return None

    def _rate(value: str | None) -> float | None:
        if not value or "/" not in value:
            return None
        num, den = value.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else None

    rotation = 0
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    return {
        "source": "ffprobe",
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "avg_frame_rate": _rate(stream.get("avg_frame_rate")),
        "r_frame_rate": _rate(stream.get("r_frame_rate")),
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "duration_s": float(data.get("format", {}).get("duration", 0)) or None,
        "bit_rate": data.get("format", {}).get("bit_rate"),
        "rotation": rotation,
        "pix_fmt": stream.get("pix_fmt"),
    }


def _probe_opencv(video: Path) -> dict[str, Any]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    finally:
        cap.release()
    return {
        "source": "opencv",
        "codec": None,
        "width": w,
        "height": h,
        "avg_frame_rate": fps,
        "r_frame_rate": fps,
        "nb_frames": n,
        "duration_s": (n / fps) if (n and fps) else None,
        "bit_rate": None,
        "rotation": 0,
        "pix_fmt": None,
    }


def probe_video(video: Path) -> dict[str, Any]:
    probe = _probe_ffprobe(video)
    if probe is None:
        probe = _probe_opencv(video)
        probe["note"] = "ffprobe unavailable, values read via OpenCV"
    return probe


def normalise_session(raw: dict) -> tuple[dict, list[str]]:
    """Fill absent optional fields with the protocol's explicit 'not provided' string."""
    out = dict(raw)
    missing = []
    for field in REQUIRED_SESSION_FIELDS:
        value = out.get(field)
        if value in (None, "", []):
            out[field] = NOT_PROVIDED
            missing.append(field)
    return out, missing


def validate_session(session: dict, schema_path: Path) -> list[str]:
    import jsonschema
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(session)]


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S0")
    video = Path(ctx.video_path)

    io_guard.assert_source_not_writable(video)
    res.check("source_not_writable", True,
              "Source directory is outside the writable whitelist.",
              evidence=str(video.parent))

    res.check("video_exists", video.is_file(), f"Source video: {video}")
    if not video.is_file():
        return res

    digest = sha256_file(video)
    probe = probe_video(video)

    schema_errors = validate_session(ctx.session, ctx.cfg.paths.schemas_dir / "session.schema.json")
    res.check("session_schema_valid", not schema_errors,
              "; ".join(schema_errors) or "session.json validates against the schema.")

    session, missing = normalise_session(ctx.session)
    res.check("session_fields_present", not missing,
              f"Recorded as '{NOT_PROVIDED}': {missing}" if missing else "All session fields supplied.",
              severity=WARN, evidence=missing)

    measured_fps = probe.get("avg_frame_rate") or probe.get("r_frame_rate")
    res.check("fps_measured", bool(measured_fps and measured_fps > 0),
              f"Measured frame rate: {measured_fps}")

    declared = ctx.session.get("declared_frame_rate")
    if declared and measured_fps:
        agrees = abs(declared - measured_fps) / measured_fps < 0.02
        res.check("declared_fps_matches_measured", agrees,
                  f"Declared {declared} fps vs measured {measured_fps:.3f} fps. "
                  f"The report uses the MEASURED value.",
                  severity=WARN)

    duration = probe.get("duration_s")
    res.check("duration_known", bool(duration and duration > 0),
              f"Duration: {duration} s", severity=WARN)

    res.check("resolution_known", bool(probe.get("width") and probe.get("height")),
              f"{probe.get('width')}x{probe.get('height')}")

    payload = {
        "run_id": ctx.run_id,
        "video_path": str(video),
        "video_sha256": digest,
        "video_bytes": video.stat().st_size,
        "probe": probe,
        "measured_fps": measured_fps,
        "session": session,
        "session_missing_fields": missing,
        "config_hash": ctx.cfg.config_hash,
        "ffmpeg": _ffmpeg_exe(),
        "ffprobe": _ffprobe_exe(),
    }
    out = ctx.write_json("00_ingest.json", payload)
    res.outputs["ingest"] = str(out)
    res.stats = {
        "video_sha256": digest[:16],
        "measured_fps": measured_fps,
        "duration_s": duration,
        "resolution": f"{probe.get('width')}x{probe.get('height')}",
    }
    return res
