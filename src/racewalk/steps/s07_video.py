"""S7 annotated video.

IN     frames/, 02_landmarks.parquet, 03_kinematics.parquet, 04_gait.json,
       04_foot_height.parquet, 05_metrics.json, 00_ingest.json
DO     render every analysed frame through the same overlay as S6 (video mode: skeleton,
       leg lines, contact markers, live joint angles, both legs' phase badges, live
       angle table, straight-leg flag, event banner), pipe to ffmpeg, then cut one clip per
       flagged event
OUT    07_video/annotated_full.mp4, 07_video/ev<id>_<TYPE>.mp4,
       outputs/<Athlete>_Annotated_<date>_vNN.mp4 (versioned, never overwritten)
VERIFY frame count and duration match the analysed frames, one clip per event, source
       video untouched

Angles in the video are live per-frame values from 03_kinematics.parquet, rounded to one
decimal. Gated values show as n/a, never interpolated. Frames are downscaled to
video.output_max_width BEFORE annotation, so a 4K phone clip costs 1080p-class time.
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from racewalk.config import sha256_file
from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.framedata import FrameData
from racewalk.io_guard import guarded_path
from racewalk.overlay import render
from racewalk.phase_defs import EVENT_DISPLAY
from racewalk.steps.s00_ingest import _ffmpeg_exe
from racewalk.versioning import next_versioned, safe

VIDEO_EVENTS = ("STRAIGHT_LEG", "CONTACT")      # short events get a per-frame banner


def _values(row) -> dict:
    return {k: round(float(v), 3 if k.endswith("_norm") or k.endswith("_s") else 1)
            for k, v in row.items() if isinstance(v, (float, np.floating)) and np.isfinite(v)}


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _live_table(vals: dict, fd: FrameData, i: int, view_class: str,
                near_side: str | None = None) -> list[tuple[str, str, bool]]:
    def dim(lg: str) -> bool:
        return view_class == "sagittal" and near_side is not None and lg != near_side

    def fmt(key, dec=1, unit="deg"):
        return "n/a" if vals.get(key) is None else f"{vals[key]:.{dec}f} {unit}"
    rows = [("Knee L", fmt("knee_left_deg"), dim("L")), ("Knee R", fmt("knee_right_deg"), dim("R")),
            ("Hip L", fmt("hip_left_deg"), dim("L")), ("Hip R", fmt("hip_right_deg"), dim("R")),
            ("Ankle L", fmt("ankle_left_deg"), dim("L")), ("Ankle R", fmt("ankle_right_deg"), dim("R")),
            ("Elbow L", fmt("elbow_left_deg"), dim("L")), ("Elbow R", fmt("elbow_right_deg"), dim("R"))]
    if view_class == "frontal":
        rows.append(("Pelvic tilt", fmt("pelvic_tilt_deg"), False))
    rows += [("Trunk lean" if view_class == "sagittal" else "Trunk tilt", fmt("trunk_inclination_deg"), False),
             ("Pelvic rotation", fmt("pelvic_yaw_deg"), False)]
    for lg in ("L", "R"):
        h = fd.foot_height(lg, i)
        rows.append((f"Foot height {lg}", "n/a" if h is None else f"{h:.3f} LL", dim(lg)))
    speed = ((fd.metrics or {}).get("gait") or {}).get("average_speed_kmh")
    rows.append(("Speed (avg)", "NOT PROVIDED" if speed is None else f"{speed:.2f} km/h", False))
    return rows


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S7")
    ingest = ctx.read_json("00_ingest.json")
    fd = FrameData.load(ctx.run_dir)
    events = fd.metrics["events"]
    view = ctx.view_class
    near_side = ctx.near_side
    track = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    screen = float(ctx.cfg.phase_rules["screening"]["straight_knee_min_deg"])
    crf = str(ctx.cfg.get("video.output_crf", 20))
    preset = str(ctx.cfg.get("video.output_preset", "medium"))
    max_w = int(ctx.cfg.get("video.output_max_width", 1920))
    ffmpeg = _ffmpeg_exe()

    vdir = guarded_path(ctx.video_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    full = vdir / "annotated_full.mp4"
    n = len(fd.frames)

    # frames that sit inside a short flagged event get a banner
    banner: dict[int, dict] = {}
    for e in events:
        if e["type"] in VIDEO_EVENTS:
            a = int(np.searchsorted(fd.kin["t_s"].to_numpy(), e["start_t_s"]))
            b = int(np.searchsorted(fd.kin["t_s"].to_numpy(), e["end_t_s"], side="right"))
            for i in range(max(0, a), min(n, b + 1)):
                banner.setdefault(i, {"id": e["id"], "display": EVENT_DISPLAY[e["type"]], "severity": e["severity"]})

    def annotate(i: int, scale: float):
        img = cv2.imread(str(fd.frames[i]))
        pts = fd.pts[i].copy()
        if scale != 1.0:
            img = cv2.resize(img, (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
            pts = pts * scale
        row = fd.kin.iloc[i].to_dict()
        vals = _values(row)
        legs = fd.legs_at(i)
        flags = []
        for lg, lab in (("L", "left"), ("R", "right")):
            v = vals.get(f"knee_{lab}_deg")
            if legs[lg]["phase"] in ("INITIAL_CONTACT", "LOADING", "MID_STANCE") and v is not None and v < screen:
                flags.append(f"{lg} KNEE BELOW SCREEN ({v:.0f} deg)")
        stride = legs["L"]["stride_id"] or legs["R"]["stride_id"]
        return render(img, pts, fd.vis[i], vals, view_class=view, legs=legs, stride_label=stride,
                      t_s=float(row["t_s"]), frame_idx=i, mode="video", track_conf=track,
                      measurements=_live_table(vals, fd, i, view, near_side), event=banner.get(i), flags=flags,
                      screen_knee_deg=screen, near_side=near_side)

    if ctx.cfg.get("video.render_full_annotated", True):
        w0 = fd.width
        scale = min(1.0, max_w / w0)
        first, _ = annotate(0, scale)
        H, W = first.shape[:2]
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", repr(fd.fps), "-i", "-",
               "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(full)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        report_every = max(1, n // 10)
        try:
            for i in range(n):
                canvas, _ = annotate(i, scale)
                if canvas.shape[:2] != (H, W):
                    canvas = cv2.resize(canvas, (W, H))
                proc.stdin.write(canvas.tobytes())
                if (i + 1) % report_every == 0 or i + 1 == n:
                    print(f"  video {i + 1:>6}/{n}  {100 * (i + 1) / n:5.1f}%", flush=True)
        finally:
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            code = proc.wait()
        res.check("ffmpeg_encode_ok", code == 0, err.strip()[:1500] or "Encoded annotated_full.mp4")
        if code != 0:
            return res

    got = _frame_count(full) if full.is_file() else 0
    res.check("full_video_frame_count", abs(got - n) <= 1, f"{got} frames encoded for {n} analysed frames")

    clips, clip_fail = [], []
    pad = float(ctx.cfg.get("video.event_clip_pad_s", 0.75))
    if ctx.cfg.get("video.render_event_clips", True):
        for e in events:
            length = e["end_t_s"] - e["start_t_s"]
            a = e["start_t_s"] - pad
            b = e["end_t_s"] + pad
            if length > 4.0:          # long windows (drift, asymmetry): a few seconds around the illustrated frame
                mid = e["frames"][-1]["frame"] / fd.fps if e["frames"] else e["start_t_s"]
                a, b = mid - 2.0, mid + 2.0
            a = max(0.0, a)
            name = f"ev{e['id']:02d}_{e['type']}.mp4"
            dst = vdir / name
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(full), "-ss", f"{a:.3f}",
                   "-t", f"{max(b - a, 0.5):.3f}", "-c:v", "libx264", "-preset", preset, "-crf", crf,
                   "-pix_fmt", "yuv420p", "-an", str(dst)]
            ok = subprocess.run(cmd, capture_output=True).returncode == 0 and dst.is_file()
            (clips if ok else clip_fail).append(name)
    res.check("event_clips_rendered", not clip_fail and len(clips) == (len(events) if ctx.cfg.get("video.render_event_clips", True) else 0),
              f"{len(clips)} of {len(events)} event clips written" + (f"; failed: {clip_fail}" if clip_fail else ""),
              severity=WARN)

    session = ingest.get("session", {})
    date = session.get("session_date")
    date = date if isinstance(date, str) and len(date) == 10 else dt.date.today().isoformat()
    stem = f"{safe(session.get('athlete_name', 'Athlete'))}_Annotated_{date.replace('-', '')}"
    published = next_versioned(guarded_path(ctx.cfg.paths.outputs_dir), stem, ".mp4")
    shutil.copyfile(full, guarded_path(published))
    res.check("versioned_copy_published", published.is_file(), f"Published {published.name}")
    res.check("source_unmodified",
              sha256_file(ingest["video_path"]) == ingest["video_sha256"] if Path(ingest["video_path"]).is_file() else True,
              "Source video hash unchanged after rendering.")

    res.outputs["annotated_full"] = str(full)
    res.outputs["published_video"] = str(published)
    res.outputs["clips_dir"] = str(vdir)
    ctx.write_json("07_video.json", {"full": str(full), "published": str(published), "published_name": published.name,
                                     "clips": clips, "fps": fd.fps, "frames": n,
                                     "note": "Live angles are per-frame values from 03_kinematics, rounded; gated values render as n/a."})
    res.stats = {"frames": n, "clips": len(clips), "published": published.name}
    return res
