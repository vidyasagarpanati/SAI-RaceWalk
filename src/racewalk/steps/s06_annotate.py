"""S6 annotate.

IN     frames/, 02_landmarks.parquet, 03_kinematics.parquet, 04_gait.json,
       04_foot_height.parquet, 05_metrics.json
DO     annotated screenshots: one per flagged event frame, plus a reference stride for each
       leg with one frame per phase
OUT    06_frames/ev<id>_<n>_<TYPE>.jpg, 06_frames/ref_<stride>_p<badge>_<PHASE>.jpg,
       06_manifest.json
VERIFY every requested frame exists; every overlay the spec requires is either drawn or
       explicitly skipped with a reason; labels do not sit on joints; files within budget

A race walking clip holds dozens of strides, so the report shows the strides that matter
instead of all of them: the frames that illustrate each flagged event, and one
representative stride per leg so the reader sees all six phases. The full video shows
everything.

Every number drawn on a key frame is the rounded value from 03_kinematics.parquet at that
frame (angles) or from 05_metrics.json (events), so image and evidence file agree.

The coaching box is interpretation, which belongs to the narrative step. S6 marks it as
pending; S10 re-renders the same frames through ``render_key_frame`` with the verified text
from S8/S9.
"""
from __future__ import annotations

import cv2
import numpy as np

from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.framedata import FrameData
from racewalk.io_guard import guarded_path
from racewalk.overlay import REQUIRED_KEY_FRAME_ELEMENTS, render
from racewalk.phase_defs import BADGE, DISPLAY, EVENT_DISPLAY, ORDER, PHASE_LOOK_FOR

PENDING_COACHING = ("[PENDING] Written by the narrative step from the verified findings, "
                    "then re-rendered into this box in the final report.")


def _fmt(v, dec=1, unit="deg"):
    return "n/a" if v is None else f"{v:.{dec}f} {unit}"


def frame_values(fd: FrameData, f: int) -> dict:
    row = fd.kin.iloc[f]
    return {c: round(float(v), 1) for c, v in row.items()
            if isinstance(v, (float, np.floating)) and np.isfinite(v)}


def measurement_rows(vals: dict, fd: FrameData, f: int, view_class: str,
                     near_side: str | None = None) -> list[tuple[str, str, bool]]:
    def dim(lg: str) -> bool:
        return view_class == "sagittal" and near_side is not None and lg != near_side

    rows = [("Knee L", _fmt(vals.get("knee_left_deg")), dim("L")), ("Knee R", _fmt(vals.get("knee_right_deg")), dim("R")),
            ("Hip L", _fmt(vals.get("hip_left_deg")), dim("L")), ("Hip R", _fmt(vals.get("hip_right_deg")), dim("R")),
            ("Ankle L", _fmt(vals.get("ankle_left_deg")), dim("L")), ("Ankle R", _fmt(vals.get("ankle_right_deg")), dim("R")),
            ("Elbow L", _fmt(vals.get("elbow_left_deg")), dim("L")), ("Elbow R", _fmt(vals.get("elbow_right_deg")), dim("R"))]
    if view_class == "frontal":
        rows += [("Pelvic tilt", _fmt(vals.get("pelvic_tilt_deg")), False), ("Shoulder tilt", _fmt(vals.get("shoulder_tilt_deg")), False)]
    rows.append(("Trunk lean" if view_class == "sagittal" else "Trunk tilt", _fmt(vals.get("trunk_inclination_deg")), False))
    rows.append(("Pelvic rotation (vs median)", _fmt(vals.get("pelvic_yaw_deg")), False))
    for lg in ("L", "R"):
        h = fd.foot_height(lg, f)
        rows.append((f"Foot height {lg}", "n/a" if h is None else f"{h:.3f} LL", dim(lg)))
    speed = ((fd.metrics or {}).get("gait") or {}).get("average_speed_kmh")
    rows.append(("Speed (avg)", "NOT PROVIDED" if speed is None else f"{speed:.2f} km/h", False))
    return rows


def _observation(spec: dict, vals: dict, event: dict | None) -> str:
    if event:
        base = (f"[MEASURED] {EVENT_DISPLAY[event['type']]} ({event['severity']}, {event['confidence']} confidence), "
                f"{event['start_t_s']}-{event['end_t_s']} s. ")
        if event["type"] == "STRAIGHT_LEG":
            return base + (f"Lowest loading-window knee angle {event['value']} deg against a screen of "
                           f"{event['threshold']:.0f} deg, across {event['n_strides']} stride(s). "
                           f"Depth-derived: a screen, not a ruling.")
        if event["type"] == "CONTACT":
            return base + (f"Both feet read as off the ground for {event['value']} ms. Pose-based screen only; "
                           f"a judge rules on the human eye.")
        if event["type"] == "ASYMMETRY":
            return base + f"Step-time symmetry index peaks at {event['value']}% in this window."
        return base + f"{event.get('worst_metric', 'drift')} drifts {event['value']}% in the adverse direction, late against early."
    ph = spec.get("phase")
    txt = f"[MEASURED] Stride {spec['stride_id']} ({spec['leg']} leg), {DISPLAY.get(ph, '')}. "
    return txt + "; ".join(f"{a} {b}" for a, b in (
        ("knee", _fmt(vals.get(f"knee_{'left' if spec['leg'] == 'L' else 'right'}_deg"))),
        ("hip", _fmt(vals.get(f"hip_{'left' if spec['leg'] == 'L' else 'right'}_deg")))))


def render_key_frame(fd: FrameData, spec: dict, view_class: str, coaching: str | None = None,
                     track_conf: float = 0.5, screen_knee_deg: float | None = None,
                     max_w: int = 1920, event: dict | None = None, near_side_param: str | None = None):
    """Render one screenshot. Shared with S10 so the final report can re-render with
    verified coaching text through the identical code path."""
    f = int(spec["frame"])
    frame = cv2.imread(str(fd.frames[f]))
    pts = fd.pts[f].copy()
    scale = 1.0
    if frame.shape[1] > max_w:
        scale = max_w / frame.shape[1]
        frame = cv2.resize(frame, (max_w, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        pts = pts * scale
    vals = frame_values(fd, f)
    legs = fd.legs_at(f)
    if spec.get("phase") and spec.get("leg"):
        legs[spec["leg"]] = dict(legs[spec["leg"]], phase=spec["phase"])
    a = max(0, f - int(0.6 * fd.fps))
    trail = fd.pts[a:f + 1, [23, 24]].mean(axis=1) * scale
    extras = {"foot_height_norm": {lg: fd.foot_height(lg, f) for lg in ("L", "R")}, "pelvis_trail": trail}
    obs = _observation(spec, vals, event)
    ph = spec.get("phase")
    coach = coaching or (PHASE_LOOK_FOR.get(ph) if spec.get("kind") == "reference" else PENDING_COACHING)
    flags = []
    if event is None and screen_knee_deg is not None and spec.get("leg"):
        v = vals.get(f"knee_{'left' if spec['leg'] == 'L' else 'right'}_deg")
        if v is not None and v < screen_knee_deg and ph in ("INITIAL_CONTACT", "LOADING", "MID_STANCE"):
            flags.append(f"KNEE BELOW SCREEN ({v:.1f} deg)")
    canvas, rep = render(frame, pts, fd.vis[f], vals, view_class=view_class, legs=legs,
                         stride_label=spec.get("stride_id"), t_s=float(fd.kin.at[f, "t_s"]), frame_idx=f,
                         mode="key", track_conf=track_conf,
                         measurements=measurement_rows(vals, fd, f, view_class, near_side_param), observation=obs,
                         coaching=coach, extras=extras,
                         event=({"id": event["id"], "display": EVENT_DISPLAY[event["type"]],
                                 "severity": event["severity"]} if event else None),
                         focus_leg=spec.get("leg"), screen_knee_deg=screen_knee_deg, flags=flags,
                         near_side=near_side_param)
    return canvas, rep


def pick_reference_strides(fd: FrameData, metrics: dict) -> dict[str, str]:
    """One representative stride per leg: middle third of the clip, no knee flag,
    the most visible landmarks, stride time closest to the leg's median."""
    flagged = {sid for e in metrics["events"] if e["type"] == "STRAIGHT_LEG" for sid in e["stride_ids"]}
    out = {}
    for lg in ("L", "R"):
        sts = [s for s in fd.gait["strides"] if s["leg"] == lg]
        if not sts:
            continue
        lo, hi = len(sts) // 3, max(len(sts) // 3 + 1, 2 * len(sts) // 3)
        pool = [s for s in sts[lo:hi] if s["id"] not in flagged and all(p["detected"] and p["key_frame"] is not None for p in s["phases"])]
        pool = pool or [s for s in sts if all(p["detected"] and p["key_frame"] is not None for p in s["phases"])]
        if not pool:
            continue
        med = float(np.median([s["stride_time_s"] for s in sts]))
        vis = lambda s: float(np.nanmean(fd.vis[s["hs_frame"]:s["next_hs_frame"]]))  # noqa: E731
        out[lg] = max(pool, key=lambda s: (round(vis(s), 2), -abs(s["stride_time_s"] - med)))["id"]
    return out


def build_specs(fd: FrameData, metrics: dict, max_per_type: int) -> list[dict]:
    specs: list[dict] = []
    by_type: dict[str, int] = {}
    ranked = sorted(metrics["events"], key=lambda e: ({"CRITICAL": 0, "MODERATE": 1, "MINOR": 2}[e["severity"]], e["start_t_s"]))
    keep = set()
    for e in ranked:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        if by_type[e["type"]] <= max_per_type:
            keep.add(e["id"])
    for e in metrics["events"]:
        if e["id"] not in keep:
            continue
        for i, fr in enumerate(e["frames"], 1):
            leg = next((s["leg"] for s in fd.gait["strides"] if s["id"] == fr.get("stride_id")), None)
            specs.append({"kind": "event", "event_id": e["id"], "frame": fr["frame"], "stride_id": fr.get("stride_id"),
                          "leg": leg, "phase": fr.get("phase"), "label": fr["label"],
                          "file": f"ev{e['id']:02d}_{i}_{e['type']}.jpg"})
    refs = pick_reference_strides(fd, metrics)
    for lg, sid in refs.items():
        st = next(s for s in fd.gait["strides"] if s["id"] == sid)
        for p in st["phases"]:
            specs.append({"kind": "reference", "event_id": None, "frame": p["key_frame"], "stride_id": sid, "leg": lg,
                          "phase": p["phase"], "label": f"{lg} leg, {DISPLAY[p['phase']]}",
                          "file": f"ref_{sid}_p{BADGE[p['phase']]}_{p['phase']}.jpg"})
    return specs


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S6")
    fd = FrameData.load(ctx.run_dir)
    out_dir = guarded_path(ctx.key_frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = int(ctx.cfg.get("render.jpeg_quality_embed", 85)) + 5
    track = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    max_w = int(ctx.cfg.get("render.key_frame_max_width", 1920))
    screen = float(ctx.cfg.phase_rules["screening"]["straight_knee_min_deg"])
    events = {e["id"]: e for e in fd.metrics["events"]}
    specs = build_specs(fd, fd.metrics, int(ctx.cfg.get("report.max_event_frames_per_type", 3)))

    manifest, coverage_gaps, overlaps = [], [], 0
    for sp in specs:
        canvas, rep = render_key_frame(fd, sp, ctx.view_class, track_conf=track, screen_knee_deg=screen, max_w=max_w,
                                       event=events.get(sp["event_id"]), near_side_param=ctx.near_side)
        path = out_dir / sp["file"]
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
        gaps = rep.missing(REQUIRED_KEY_FRAME_ELEMENTS)
        if gaps:
            coverage_gaps.append(f"{sp['file']}: {gaps}")
        overlaps += rep.label_joint_overlaps
        manifest.append({**sp, "t_s": round(float(fd.kin.at[int(sp["frame"]), "t_s"]), 3), "file": str(path),
                         "bytes": path.stat().st_size, "width": canvas.shape[1], "height": canvas.shape[0],
                         "elements_drawn": sorted(rep.drawn), "elements_skipped": rep.skipped,
                         "n_labels": rep.n_labels, "label_joint_overlaps": rep.label_joint_overlaps,
                         "coaching_text": "pending" if sp["kind"] == "event" else "static phase guidance"})

    out = ctx.write_json("06_manifest.json", {
        "frames": manifest, "required_elements": REQUIRED_KEY_FRAME_ELEMENTS,
        "reference_strides": pick_reference_strides(fd, fd.metrics),
        "note": "Events beyond report.max_event_frames_per_type per type get no screenshot; the video shows them all."})
    n_ev = sum(1 for m in manifest if m["kind"] == "event")
    n_ref = sum(1 for m in manifest if m["kind"] == "reference")
    res.check("key_frames_rendered", bool(manifest), f"{len(manifest)} annotated frames ({n_ev} event, {n_ref} reference)")
    res.check("reference_strides_found", n_ref >= 6,
              f"{n_ref} reference frames." if n_ref >= 6 else
              "No stride had all six phases with usable key frames, so no reference stride can be shown.", severity=WARN)
    res.check("required_overlays_accounted_for", not coverage_gaps,
              "; ".join(coverage_gaps[:4]) if coverage_gaps else
              "Every required overlay is drawn or skipped with a stated reason on every frame.")
    skipped = sorted({k for m in manifest for k in m["elements_skipped"]})
    res.check("no_overlays_skipped", not skipped,
              f"Skipped on some frames (landmarks below confidence): {skipped}" if skipped else "Nothing skipped.",
              severity=WARN)
    res.check("labels_clear_of_joints", overlaps == 0, f"{overlaps} label(s) had to be placed over a landmark.",
              severity=WARN)
    big = [m["file"] for m in manifest if m["bytes"] > 3_000_000]
    res.check("frame_size_budget", not big, f"Over 3 MB: {big}", severity=WARN)
    res.outputs["key_frames_dir"] = str(out_dir)
    res.outputs["manifest"] = str(out)
    res.stats = {"n_frames": len(manifest), "event_frames": n_ev, "reference_frames": n_ref,
                 "label_joint_overlaps": overlaps}
    return res
