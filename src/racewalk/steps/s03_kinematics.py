"""S3 kinematics.

IN     02_landmarks.parquet, 02_pose_quality.json, 01_frames.json
DO     joint angles, trunk and pelvis orientation, rotation excursions, foot position
       signals, arm-swing and shoulder signals
OUT    03_kinematics.parquet, 03_quality.json
VERIFY every measure gated on landmark confidence, NaN rate reported per measure,
       anatomical plausibility bounds asserted, declared camera view cross-checked
       against the observed body geometry

Coordinate policy, stated once and applied everywhere:
  - True joint angles (knee, hip, ankle, elbow, arm swing) are computed in MediaPipe
    WORLD space, metres, because image space distorts them with perspective. This is
    why the straight-leg SCREEN works from a rear view at all. It is still a screen:
    only a side view lets a judge, or this pipeline, rule on it.
  - Orientation against gravity or the image frame (pelvic obliquity, shoulder tilt,
    trunk inclination) is computed in image PIXEL space, because that is where the
    camera's vertical lives. Normalised image coordinates are scaled by the real frame
    width and height first, so a 16:9 frame does not distort angles.
  - Rotation of pelvis and shoulders about the vertical axis is a WORLD-space yaw
    excursion around the athlete's own median. Depth is the weakest output of
    monocular pose, so this is reported as an inferred range, not an orientation.
  - Foot height above ground is NOT computed here. It needs the gait-rule thresholds,
    so S4 derives it from the raw foot positions exported below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from racewalk import geometry as G
from racewalk import landmarks as L
from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.framedata import frame_size
from racewalk.io_guard import guarded_path

# Anatomical plausibility bounds. A value outside these is counted and reported,
# because a systematic excursion means the pose is wrong, not the athlete.
PLAUSIBLE = {
    "knee": (0.0, 190.0), "hip": (0.0, 190.0), "ankle": (0.0, 190.0),
    "elbow": (0.0, 190.0), "shoulder_arm": (0.0, 190.0),
    "trunk_inclination": (-60.0, 60.0), "pelvic_tilt": (-45.0, 45.0),
    "shoulder_tilt": (-45.0, 45.0), "pelvic_yaw": (-90.0, 90.0), "shoulder_yaw": (-90.0, 90.0),
    "torso_twist": (-90.0, 90.0), "rearfoot_dev": (0.0, 60.0),
}
_PLAUSIBLE_KEYS = sorted(PLAUSIBLE, key=len, reverse=True)


def _bounds_for(measure: str):
    for key in _PLAUSIBLE_KEYS:
        if measure.startswith(key):
            return PLAUSIBLE[key]
    return None


GATE_OK = "OK"
GATE_NO_DETECTION = "NO_DETECTION"
GATE_INSUFFICIENT = "INSUFFICIENT_LANDMARKS"
GATE_LOW_CONFIDENCE = "LOW_CONFIDENCE"
GATE_VIEW = "NOT_MEASURABLE_FROM_THIS_VIEW"


def _load_arrays(ctx: Context):
    df = pd.read_parquet(ctx.artefact("02_landmarks.parquet"))
    n = int(df["frame"].max()) + 1
    img = df[["x", "y"]].to_numpy(np.float64).reshape(n, 33, 2)
    world = df[["wx", "wy", "wz"]].to_numpy(np.float64).reshape(n, 33, 3)
    vis = df["visibility"].to_numpy(np.float64).reshape(n, 33)
    t_ms = df["t_ms"].to_numpy(np.float64).reshape(n, 33)[:, 0]
    return img, world, vis, t_ms


def _gate(vis, ids, min_conf, frame_ok):
    conf_ok = np.all(vis[:, ids] >= min_conf, axis=1)
    detected = np.isfinite(vis[:, ids]).all(axis=1)
    usable = conf_ok & frame_ok & detected
    reason = np.where(~detected, GATE_NO_DETECTION,
             np.where(~frame_ok, GATE_INSUFFICIENT,
             np.where(~conf_ok, GATE_LOW_CONFIDENCE, GATE_OK)))
    return usable, reason


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S3")
    frames_meta = ctx.read_json("01_frames.json")
    pose_quality = ctx.read_json("02_pose_quality.json")
    fps = float(frames_meta["analysis_fps"])
    dt = 1.0 / fps
    W, H = frame_size(ctx.run_dir)

    img, world, vis, t_ms = _load_arrays(ctx)
    n = img.shape[0]
    px = img * np.array([W, H], dtype=np.float64)

    min_conf = float(ctx.cfg.get("quality_gates.angle_min_confidence", 0.4))
    min_visible = int(ctx.cfg.get("quality_gates.min_visible_landmarks", 20))
    track_conf = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    n_visible = np.nansum(vis >= track_conf, axis=1)
    frame_ok = n_visible >= min_visible
    ID = L.ID
    view = ctx.view_class

    out: dict[str, np.ndarray] = {
        "frame": np.arange(n), "t_ms": t_ms, "t_s": t_ms / 1000.0,
        "n_visible_landmarks": n_visible, "frame_usable": frame_ok,
    }
    reasons: dict[str, np.ndarray] = {}
    gated: list[str] = []

    def register(name, values, usable, reason):
        out[name] = np.where(usable, values, np.nan)
        reasons[name] = reason
        gated.append(name)

    def add_angle(name, vertex, p1, p2, space="world"):
        usable, reason = _gate(vis, [vertex, p1, p2], min_conf, frame_ok)
        src = world if space == "world" else px
        register(name, G.angle_at(src[:, vertex], src[:, p1], src[:, p2]), usable, reason)

    # -- true joint angles, world space -----------------------------------------
    for label, prefix in (("left", "LEFT"), ("right", "RIGHT")):
        s = L.side(prefix)
        add_angle(f"knee_{label}_deg", s["knee"], s["hip"], s["ankle"])
        add_angle(f"hip_{label}_deg", s["hip"], s["shoulder"], s["knee"])
        add_angle(f"ankle_{label}_deg", s["ankle"], s["knee"], s["foot"])
        add_angle(f"elbow_{label}_deg", s["elbow"], s["shoulder"], s["wrist"])
        add_angle(f"shoulder_arm_{label}_deg", s["shoulder"], s["elbow"], s["hip"])

    # -- orientation, image pixel space -----------------------------------------
    sh_l, sh_r = ID["LEFT_SHOULDER"], ID["RIGHT_SHOULDER"]
    hip_l, hip_r = ID["LEFT_HIP"], ID["RIGHT_HIP"]
    shoulder_mid = (px[:, sh_l] + px[:, sh_r]) / 2.0
    pelvis_mid = (px[:, hip_l] + px[:, hip_r]) / 2.0

    u, r = _gate(vis, [hip_l, hip_r], min_conf, frame_ok)
    register("pelvic_tilt_deg", G.signed_tilt(px[:, hip_l], px[:, hip_r]), u, r)
    u, r = _gate(vis, [sh_l, sh_r], min_conf, frame_ok)
    register("shoulder_tilt_deg", G.signed_tilt(px[:, sh_l], px[:, sh_r]), u, r)

    # Facing direction (sagittal views): which way the athlete travels in the image.
    nose_dx = px[:, ID["NOSE"], 0] - pelvis_mid[:, 0]
    facing = 1.0 if np.nanmedian(nose_dx) >= 0 else -1.0
    u, r = _gate(vis, [sh_l, sh_r, hip_l, hip_r], min_conf, frame_ok)
    trunk = G.line_angle_from_vertical(shoulder_mid, pelvis_mid)
    if view == "sagittal":
        trunk = trunk * facing     # positive = leaning forward, the way the athlete travels
    register("trunk_inclination_deg", trunk, u, r)

    # -- rotation excursions, world space --------------------------------------
    u, r = _gate(vis, [hip_l, hip_r], min_conf, frame_ok)
    py = G.unwrap_centered(np.where(u, G.yaw_from_pair(world[:, hip_l], world[:, hip_r]), np.nan))
    register("pelvic_yaw_deg", py, u, r)
    u2, r2 = _gate(vis, [sh_l, sh_r], min_conf, frame_ok)
    sy = G.unwrap_centered(np.where(u2, G.yaw_from_pair(world[:, sh_l], world[:, sh_r]), np.nan))
    register("shoulder_yaw_deg", sy, u2, r2)
    both = u & u2
    register("torso_twist_deg", out["shoulder_yaw_deg"] - out["pelvic_yaw_deg"], both,
             np.where(~u, r, r2))

    # -- shoulder elevation (tension proxy) -------------------------------------
    elev = []
    for prefix in ("LEFT", "RIGHT"):
        s = L.side(prefix)
        elev.append(G.distance(px[:, s["ear"]], px[:, s["shoulder"]]))
    sw_px = G.distance(px[:, sh_l], px[:, sh_r])
    sw_px = np.where(sw_px > 1.0, sw_px, np.nan)
    u, r = _gate(vis, [ID["LEFT_EAR"], ID["RIGHT_EAR"], sh_l, sh_r], min_conf, frame_ok)
    register("shoulder_elevation_norm", np.mean(elev, axis=0) / sw_px, u, r)

    # -- rear-foot alignment (frontal views only) ------------------------------
    for label, prefix in (("left", "LEFT"), ("right", "RIGHT")):
        s = L.side(prefix)
        if view == "frontal":
            u, r = _gate(vis, [s["knee"], s["ankle"], s["heel"]], min_conf, frame_ok)
            dev = 180.0 - G.angle_at(px[:, s["ankle"]], px[:, s["knee"]], px[:, s["heel"]])
            register(f"rearfoot_dev_{label}_deg", dev, u, r)
        else:
            out[f"rearfoot_dev_{label}_deg"] = np.full(n, np.nan)
            reasons[f"rearfoot_dev_{label}_deg"] = np.full(n, GATE_VIEW)
            gated.append(f"rearfoot_dev_{label}_deg")

    # -- foot pitch (sagittal views only) ---------------------------------------
    for label, prefix in (("left", "LEFT"), ("right", "RIGHT")):
        s = L.side(prefix)
        if view == "sagittal":
            u, r = _gate(vis, [s["heel"], s["foot"]], min_conf, frame_ok)
            d = px[:, s["foot"]] - px[:, s["heel"]]
            # positive = toe below heel (flat or forefoot), negative = toe up (heel first)
            pitch = np.degrees(np.arctan2(d[:, 1], np.abs(d[:, 0]) + G.EPS))
            register(f"foot_pitch_{label}_deg", pitch, u, r)
        else:
            out[f"foot_pitch_{label}_deg"] = np.full(n, np.nan)
            reasons[f"foot_pitch_{label}_deg"] = np.full(n, GATE_VIEW)
            gated.append(f"foot_pitch_{label}_deg")

    # -- scale ----------------------------------------------------------------
    out["shoulder_width_px"] = sw_px
    out["hip_width_px"] = G.distance(px[:, hip_l], px[:, hip_r])
    seg = []
    for prefix in ("LEFT", "RIGHT"):
        s = L.side(prefix)
        seg.append(G.distance(px[:, s["hip"]], px[:, s["knee"]]) + G.distance(px[:, s["knee"]], px[:, s["ankle"]]))
    leg_len = float(np.nanmedian(np.concatenate(seg)))
    out["leg_length_px"] = np.full(n, leg_len)

    # -- raw foot and pelvis positions, smoothed lightly ------------------------
    win_f = int(ctx.cfg.get("smoothing.foot_window", 5))
    ord_p = int(ctx.cfg.get("smoothing.polyorder", 2))
    for label, prefix in (("left", "LEFT"), ("right", "RIGHT")):
        s = L.side(prefix)
        u, r = _gate(vis, [s["heel"], s["foot"], s["ankle"]], min_conf, frame_ok)
        low = np.maximum(px[:, s["heel"], 1], px[:, s["foot"], 1])   # y grows downward
        out[f"foot_low_y_{label}_px"] = G.smooth_nan(np.where(u, low, np.nan), win_f, ord_p)
        reasons[f"foot_low_y_{label}_px"] = r
        gated.append(f"foot_low_y_{label}_px")
        out[f"ankle_x_{label}_px"] = G.smooth_nan(np.where(u, px[:, s["ankle"], 0], np.nan), win_f, ord_p)
        out[f"heel_y_{label}_px"] = G.smooth_nan(np.where(u, px[:, s["heel"], 1], np.nan), win_f, ord_p)
        out[f"foot_len_{label}_px"] = np.where(u, G.distance(px[:, s["heel"]], px[:, s["foot"]]), np.nan)
    u, r = _gate(vis, [hip_l, hip_r], min_conf, frame_ok)
    out["pelvis_x_px"] = G.smooth_nan(np.where(u, pelvis_mid[:, 0], np.nan), win_f, ord_p)
    out["pelvis_y_px"] = G.smooth_nan(np.where(u, pelvis_mid[:, 1], np.nan), win_f, ord_p)
    out["pelvis_y_leg"] = out["pelvis_y_px"] / leg_len          # vertical oscillation, in leg lengths
    reasons["pelvis_y_leg"] = r
    gated.append("pelvis_y_leg")

    # -- smoothing of the angle and orientation columns -----------------------
    smoothing = bool(ctx.cfg.get("smoothing.enabled", True))
    window = int(ctx.cfg.get("smoothing.window", 9))
    polyorder = int(ctx.cfg.get("smoothing.polyorder", 2))
    skip = {"frame", "t_ms", "t_s", "n_visible_landmarks", "frame_usable", "leg_length_px",
            "shoulder_width_px", "hip_width_px", "pelvis_y_leg", "pelvis_x_px", "pelvis_y_px"}
    skip |= {c for c in out if c.startswith(("foot_low_y", "ankle_x_", "heel_y_", "foot_len_"))}
    smoothed_cols = []
    if smoothing and str(ctx.cfg.get("smoothing.method", "savgol")) == "savgol":
        for name in list(out.keys()):
            if name in skip:
                continue
            out[name] = G.smooth_nan(out[name], window, polyorder)
            smoothed_cols.append(name)

    df_out = pd.DataFrame(out)
    out_parquet = guarded_path(ctx.artefact("03_kinematics.parquet"))
    df_out.to_parquet(out_parquet, index=False)

    # -- quality accounting -------------------------------------------------------
    nan_rates, dominant = {}, {}
    for name in gated:
        col = out[name]
        nan_rates[name] = float(np.mean(~np.isfinite(col)))
        if name in reasons:
            bad = reasons[name][~np.isfinite(col)]
            if bad.size:
                vals, cnts = np.unique(bad, return_counts=True)
                dominant[name] = str(vals[int(np.argmax(cnts))])
    implausible = {}
    for name in gated:
        b = _bounds_for(name)
        if b is None:
            continue
        finite = out[name][np.isfinite(out[name])]
        if finite.size:
            bad = int(np.sum((finite < b[0]) | (finite > b[1])))
            if bad:
                implausible[name] = {"count": bad, "share": bad / finite.size, "bounds": list(b)}

    # -- view cross-check ------------------------------------------------------
    torso = G.distance(shoulder_mid, pelvis_mid)
    ratio = float(np.nanmedian(sw_px / torso)) if np.isfinite(torso).any() else float("nan")
    view_ok, view_detail = True, "Not assessable: shoulders or hips not measurable."
    if np.isfinite(ratio):
        if view == "frontal":
            view_ok = ratio >= 0.45
        elif view == "sagittal":
            view_ok = ratio <= 0.75
        view_detail = (f"Shoulder width is {ratio:.2f} of the shoulder-to-hip distance in the image. "
                       f"session.json says camera_view={ctx.camera_view!r} ({view}). A frontal (rear or front) "
                       f"view normally reads 0.45 or more and a side view 0.75 or less."
                       + ("" if view_ok else " The geometry disagrees with the setting. Check camera_view."))

    quality = {
        "n_frames": n, "analysis_fps": fps, "frame_px": [W, H],
        "camera_view": ctx.camera_view, "view_class": view,
        "facing_sign": facing if view == "sagittal" else None,
        "leg_length_px": round(leg_len, 1),
        "coordinate_policy": {
            "joint_angles": "MediaPipe world landmarks (metres, hip-centred)",
            "orientation_measures": "image pixel space (camera vertical)",
            "rotation": "world-space yaw excursion around the athlete's own median (inferred)",
            "distances": "normalised by leg length or shoulder width",
        },
        "gates": {"angle_min_confidence": min_conf, "min_visible_landmarks": min_visible},
        "smoothing": {"enabled": smoothing, "window": window, "foot_window": win_f,
                      "polyorder": polyorder, "columns": smoothed_cols},
        "nan_rate_by_measure": nan_rates, "dominant_gate_reason": dominant,
        "implausible_values": implausible,
        "usable_frame_share": float(np.mean(frame_ok)),
        "pose_detection_rate": pose_quality.get("detection_rate"),
        "view_check": {"ok": bool(view_ok), "detail": view_detail, "ratio": None if not np.isfinite(ratio) else round(ratio, 3)},
    }
    out_quality = ctx.write_json("03_quality.json", quality)

    core = ["knee_left_deg", "knee_right_deg", "pelvic_tilt_deg", "trunk_inclination_deg",
            "foot_low_y_left_px", "foot_low_y_right_px"]
    worst = max((nan_rates.get(c, 1.0) for c in core), default=1.0)
    res.check("kinematics_written", out_parquet.is_file(), str(out_parquet))
    res.check("core_measures_available", worst <= 0.40,
              f"Worst NaN rate among the core measures is {worst:.1%}. Above 40% the walk cannot be "
              f"measured reliably. Per-measure rates: "
              + ", ".join(f"{c}={nan_rates.get(c, 1.0):.0%}" for c in core))
    res.check("no_systematic_implausible_values",
              all(v["share"] < 0.05 for v in implausible.values()),
              f"Angles outside anatomical bounds: {implausible}" if implausible
              else "All angles within anatomical plausibility bounds.")
    res.check("foot_signals_present",
              bool(np.isfinite(out["foot_low_y_left_px"]).any() and np.isfinite(out["foot_low_y_right_px"]).any()),
              "Both feet have a position signal, so gait events can be detected.")
    res.check("camera_view_consistent_with_geometry", view_ok, view_detail, severity=WARN)
    res.check("leg_length_scale_valid", np.isfinite(leg_len) and leg_len > 20,
              f"Leg length is {leg_len:.0f} px in the frame. Feet heights are normalised by it.")
    res.outputs["kinematics"] = str(out_parquet)
    res.outputs["kinematics_quality"] = str(out_quality)
    res.stats = {"n_frames": n, "usable_frame_share": round(float(np.mean(frame_ok)), 4),
                 "worst_core_nan_rate": round(worst, 4), "n_measures": len(gated),
                 "view_class": view}
    return res
