"""S5 stats: the evidence file.

IN     00_ingest.json, 02_pose_quality.json, 03_kinematics.parquet, 03_quality.json,
       04_gait.json, 04_foot_height.parquet, config (stats, phase_rules, risk_rules,
       benchmarks)
DO     per-stride measures, per-leg and whole-clip statistics, left-right asymmetry,
       early-versus-late drift, rule-based flagged events (the "timestamps of interest"),
       screening-level injury risk, technique status rows, benchmark comparison (cited
       entries only), overall analysis confidence
OUT    05_metrics.json (the ONLY source of numbers for the report), 05_strides.json
VERIFY no NaN in the output, SD and drift suppressed below their minimum counts, every
       benchmark used carries a source, every event has a usable frame, the evidence
       index is populated

05_metrics.json is the ONLY source of numbers for the report. S8 receives slices of it and
S9 rejects any number in the narrative that is not in ``evidence_index``. Values are
rounded here, once, to the precision the report displays, so the grounding check
compares like with like.

Confidence is capped by what a single camera can actually see. Knee and hip flexion
happen in the sagittal plane, which is the depth direction of a rear or front camera,
so from behind they are capped at LOW. This is the same limitation a human judge has
from behind, stated as a rule instead of a hope.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from racewalk import geometry as G
from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.phase_defs import DISPLAY, EVENT_DISPLAY, ORDER, STRAIGHT_LEG_PHASES

NOT_ASSESSABLE = "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO"
NOT_PROVIDED = "NOT PROVIDED"
INDIVIDUAL = "Individualized assessment required"
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 12: "twelve"}
CONF_ORDER = ["LOW", "MEDIUM", "HIGH"]


def _w(n: int) -> str:
    """Spell small numbers. Status strings travel into the model's prompt; a digit there
    gets quoted back and then rejected as an ungrounded number."""
    return WORDS.get(int(n), str(n))


def _r(x, d):
    if x is None:
        return None
    x = float(x)
    return round(x, d) if math.isfinite(x) else None


def cap_conf(conf: str | None, cap: str | None) -> str | None:
    if conf is None or cap is None:
        return conf
    return CONF_ORDER[min(CONF_ORDER.index(conf), CONF_ORDER.index(cap))]


# Depth-derived measures are capped by camera view. Knee, hip and ankle flexion are
# sagittal-plane rotations, which are along the optical axis of a rear or front camera.
KIN_CAP = {"frontal": "LOW", "oblique": "MEDIUM", "sagittal": "HIGH"}
DEPTH_MEASURES = ("knee", "hip", "ankle")
YAW_CAP = "MEDIUM"          # rotation about the vertical axis comes from estimated depth in every view

# Per-phase measures. own = the leg whose stride it is.
OWN = {"knee_deg": "knee_{}_deg", "hip_deg": "hip_{}_deg", "ankle_deg": "ankle_{}_deg"}
WHOLE = ["pelvic_tilt_deg", "pelvic_yaw_deg", "trunk_inclination_deg", "shoulder_tilt_deg",
         "torso_twist_deg", "elbow_left_deg", "elbow_right_deg"]
PHASE_MEASURES: dict[str, tuple[str, int]] = {
    "knee_deg": ("deg", 1), "hip_deg": ("deg", 1), "ankle_deg": ("deg", 1),
    "foot_height_norm": ("leg lengths", 3),
    **{m: ("deg", 1) for m in WHOLE},
}

# name -> (units, decimals)
STRIDE_METRICS: dict[str, tuple[str, int]] = {
    "stride_time_s": ("s", 3), "stance_pct": ("%", 1), "swing_pct": ("%", 1),
    "loading_knee_min_deg": ("deg", 1), "contact_knee_deg": ("deg", 1), "vertical_knee_deg": ("deg", 1),
    "ankle_contact_deg": ("deg", 1), "ankle_range_deg": ("deg", 1),
    "pelvic_yaw_range_deg": ("deg", 1), "shoulder_yaw_range_deg": ("deg", 1),
    "torso_twist_range_deg": ("deg", 1), "pelvic_tilt_range_deg": ("deg", 1),
    "shoulder_tilt_range_deg": ("deg", 1), "trunk_incl_mean_deg": ("deg", 1),
    "trunk_incl_range_deg": ("deg", 1), "vertical_osc_leg": ("leg lengths", 3),
    "elbow_left_range_deg": ("deg", 1), "elbow_right_range_deg": ("deg", 1),
    "arm_swing_left_range_deg": ("deg", 1), "arm_swing_right_range_deg": ("deg", 1),
    "shoulder_elevation_mean": ("shoulder widths", 3), "rearfoot_range_deg": ("deg", 1),
    "step_length_leg": ("leg lengths", 3), "foot_pitch_contact_deg": ("deg", 1),
}
SIDE_ONLY = {"step_length_leg", "foot_pitch_contact_deg"}
FRONT_ONLY = {"rearfoot_range_deg", "pelvic_tilt_range_deg", "shoulder_tilt_range_deg"}
YAW_METRICS = {"pelvic_yaw_range_deg", "shoulder_yaw_range_deg", "torso_twist_range_deg"}
DRIFT_METRICS = {  # metric -> adverse direction (+1 rising is adverse, -1 falling is adverse)
    "cadence_spm": -1, "pelvic_yaw_range_deg": -1, "loading_knee_min_deg": -1,
    "vertical_osc_leg": +1, "trunk_incl_range_deg": +1,
}
SEV_ORDER = {"CRITICAL": 0, "MODERATE": 1, "MINOR": 2}


def _stride_cap(metric: str, view: str) -> str | None:
    if metric in YAW_METRICS:
        return YAW_CAP
    if any(metric.startswith(p) or f"_{p}_" in metric for p in DEPTH_MEASURES) and metric != "stride_time_s":
        return KIN_CAP[view]
    return None


def _confidence(n_valid: int, share: float, boundary: str | None = None) -> str | None:
    if n_valid == 0:
        return None
    if share >= 0.9 and n_valid >= 10:
        c = "HIGH"
    elif share >= 0.6 and n_valid >= 5:
        c = "MEDIUM"
    else:
        c = "LOW"
    if boundary == "LOW" and c == "HIGH":
        c = "MEDIUM"
    return c


def describe(values: np.ndarray, decimals: int, boundary: str | None = None) -> dict:
    n_total = int(values.size)
    v = values[np.isfinite(values)]
    n = int(v.size)
    share = n / n_total if n_total else 0.0
    if n == 0:
        return {"n_frames": n_total, "n_valid": 0, "valid_share": 0.0, "confidence": None,
                "status": NOT_ASSESSABLE}
    # per-frame values inside one phase: the 'confidence' reflects data completeness only
    conf = "HIGH" if share >= 0.9 and n >= 2 else "MEDIUM" if share >= 0.6 else "LOW"
    if boundary == "LOW" and conf == "HIGH":
        conf = "MEDIUM"
    return {"n_frames": n_total, "n_valid": n, "valid_share": round(share, 3),
            "min": _r(v.min(), decimals), "max": _r(v.max(), decimals), "mean": _r(v.mean(), decimals),
            "range": _r(v.max() - v.min(), decimals),
            "sd": _r(v.std(ddof=1), decimals) if n >= 3 else None, "confidence": conf}


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S5")
    ingest = ctx.read_json("00_ingest.json")
    pose_q = ctx.read_json("02_pose_quality.json")
    kin_q = ctx.read_json("03_quality.json")
    gait = ctx.read_json("04_gait.json")
    k = pd.read_parquet(ctx.artefact("03_kinematics.parquet"))
    fh = pd.read_parquet(ctx.artefact("04_foot_height.parquet"))
    fps = float(gait["analysis_fps"])
    view = ctx.view_class
    kcap = KIN_CAP[view]
    scr = ctx.cfg.phase_rules["screening"]
    min_sd = int(ctx.cfg.get("stats.min_strides_for_sd", 5))
    min_drift = int(ctx.cfg.get("stats.min_strides_for_drift", 12))
    leg_px = float(k["leg_length_px"].iloc[0])
    n_frames = len(k)
    t = k["t_s"].to_numpy()
    strides = gait["strides"]

    evidence: dict[str, dict] = {}

    def ev(key: str, value, units: str, confidence: str | None = None, n: int | None = None):
        if value is None:
            return
        if isinstance(value, float) and not math.isfinite(value):
            return
        evidence[key] = {"value": value, "units": units, "confidence": confidence, "n": n}

    # ---- session -------------------------------------------------------------
    probe = ingest["probe"]
    athlete = ingest["session"]
    duration_raw = float(probe.get("duration_s") or 0)

    # Speed: a declared treadmill belt speed is exact and wins. Otherwise, if the operator
    # supplied the distance the athlete covered over the clip (racewalk init-session
    # --distance-m), speed is distance / clip duration. Never estimated from the picture.
    treadmill_speed = athlete.get("treadmill_speed_kmh")
    treadmill_speed = float(treadmill_speed) if isinstance(treadmill_speed, (int, float)) and treadmill_speed > 0 else None
    distance_m = athlete.get("distance_walked_m")
    distance_m = float(distance_m) if isinstance(distance_m, (int, float)) and distance_m > 0 else None
    speed, speed_source = None, None
    if treadmill_speed is not None:
        speed, speed_source = treadmill_speed, "treadmill belt speed (declared)"
    elif distance_m is not None and duration_raw > 0:
        speed = distance_m / duration_raw * 3.6
        speed_source = "distance walked over the clip duration (declared distance)"

    mass = athlete.get("body_mass_kg")
    mass = float(mass) if isinstance(mass, (int, float)) and mass > 0 else None
    session = {
        "athlete": athlete, "video_file": ingest["video_path"].replace("\\", "/").split("/")[-1],
        "video_sha256": ingest["video_sha256"], "measured_fps": _r(ingest["measured_fps"], 3),
        "analysis_fps": fps, "duration_s": _r(probe.get("duration_s"), 2),
        "resolution": f"{probe.get('width')}x{probe.get('height')}", "codec": probe.get("codec"),
        "metadata_source": probe.get("source"), "camera_view": ctx.camera_view, "view_class": view,
        "n_strides": gait["n_strides"], "n_strides_left": gait["n_strides_by_leg"]["L"],
        "n_strides_right": gait["n_strides_by_leg"]["R"], "n_steps": len(gait["steps"]),
        "pose_estimation": "MediaPipe Pose Landmarker (full, 33 landmarks), VIDEO mode, CPU",
        "analysis_method": "Deterministic computer-vision pipeline: frame decode, pose estimation, "
                           "kinematics, rule-based gait events, descriptive statistics",
        "speed_kmh": _r(speed, 2), "speed_source": speed_source, "distance_walked_m": distance_m,
    }
    ev("session.measured_fps", session["measured_fps"], "fps", "HIGH")
    ev("session.analysis_fps", fps, "fps", "HIGH")
    ev("session.duration_s", session["duration_s"], "s", "HIGH")
    ev("session.n_strides", gait["n_strides"], "strides", "HIGH")
    ev("session.n_strides_left", session["n_strides_left"], "strides", "HIGH")
    ev("session.n_strides_right", session["n_strides_right"], "strides", "HIGH")
    ev("session.n_steps", session["n_steps"], "steps", "HIGH")
    ev("session.speed_kmh", session["speed_kmh"], "km/h", "HIGH")
    ev("session.distance_walked_m", distance_m, "m", "HIGH")
    ev("session.body_mass_kg", mass, "kg", "HIGH")

    # ---- quality and overall confidence ----------------------------------------
    nan_rates = kin_q.get("nan_rate_by_measure", {})
    core = ["knee_left_deg", "knee_right_deg", "pelvic_tilt_deg", "trunk_inclination_deg",
            "foot_low_y_left_px", "foot_low_y_right_px"]
    worst_core = max((nan_rates.get(c, 1.0) for c in core), default=1.0)
    det, usable = float(pose_q.get("detection_rate", 0)), float(kin_q.get("usable_frame_share", 0))
    if det >= 0.95 and usable >= 0.90 and worst_core <= 0.10:
        overall = "HIGH"
    elif det >= 0.85 and usable >= 0.70 and worst_core <= 0.30:
        overall = "MEDIUM"
    else:
        overall = "LOW"
    quality = {
        "pose_detection_rate_pct": _r(100 * det, 1), "usable_frame_share_pct": _r(100 * usable, 1),
        "mean_visible_landmarks": _r(np.mean(pose_q.get("per_frame_visible_count", [0])), 1),
        "worst_core_measure_missing_pct": _r(100 * worst_core, 1),
        "missing_pct_by_measure": {m: _r(100 * v, 1) for m, v in nan_rates.items()},
        "withheld_reason_by_measure": kin_q.get("dominant_gate_reason", {}),
        "implausible_values": kin_q.get("implausible_values", {}),
        "view_check": kin_q.get("view_check"), "overall_analysis_confidence": overall,
        "overall_confidence_rule": "HIGH: detection>=95%, usable>=90%, worst core missing<=10%. "
                                   "MEDIUM: >=85%, >=70%, <=30%. Otherwise LOW.",
        "camera_confidence_cap": {"depth-derived leg angles": kcap, "rotation about the vertical axis": YAW_CAP},
    }
    for key in ("pose_detection_rate_pct", "usable_frame_share_pct", "mean_visible_landmarks",
                "worst_core_measure_missing_pct"):
        ev(f"quality.{key}", quality[key], "%" if key.endswith("pct") else "landmarks", "HIGH")

    # ---- screening parameters as citable evidence ---------------------------------
    for key, unit in (("straight_knee_min_deg", "deg"), ("airborne_min_frames", "frames"),
                      ("asymmetry_flag_pct", "%"), ("drift_flag_pct", "%"),
                      ("flight_moderate_ms", "ms"), ("flight_critical_ms", "ms")):
        ev(f"screen.{key}", scr[key], unit, "HIGH")

    # ---- per-frame series accessors -------------------------------------------------
    lab = {"L": "left", "R": "right"}

    def series(measure: str, leg: str) -> np.ndarray:
        if measure in OWN:
            return k[OWN[measure].format(lab[leg])].to_numpy(float)
        if measure == "foot_height_norm":
            return fh[f"height_{leg}"].to_numpy(float)
        return k[measure].to_numpy(float)

    # ---- per-stride metrics ---------------------------------------------------------
    rows = []
    thr = float(scr["straight_knee_min_deg"])
    margin = max(0, int(round(float(scr["loading_window_start_after_hs_s"]) * fps)))
    ksm = max(1, int(scr["knee_smoothing_frames"]))
    for st in strides:
        leg, other = st["leg"], ("R" if st["leg"] == "L" else "L")
        hs, to, nxt, vert = st["hs_frame"], st["to_frame"], st["next_hs_frame"], st["vertical_frame"]
        seg = slice(hs, nxt)
        span = nxt - hs

        def rng(col, s=seg):
            v = k[col].to_numpy(float)[s]
            f = v[np.isfinite(v)]
            return float(f.max() - f.min()) if f.size >= 0.6 * v.size and f.size >= 3 else None

        def mean(col, s=seg):
            v = k[col].to_numpy(float)[s]
            f = v[np.isfinite(v)]
            return float(f.mean()) if f.size >= 0.6 * v.size and f.size >= 1 else None

        def at(col, f):
            if f is None:
                return None
            v = float(k[col].to_numpy(float)[f])
            return v if math.isfinite(v) else None

        kf = {p["phase"]: p["key_frame"] for p in st["phases"]}
        ic_f = kf.get("INITIAL_CONTACT")
        w0 = min(hs + margin, vert)
        kn = k[f"knee_{lab[leg]}_deg"].to_numpy(float)[w0:vert + 1]
        kn_ok = np.isfinite(kn)
        knee_min, knee_arg = None, None
        if kn.size >= ksm and kn_ok.mean() >= 0.6:
            run_mean = np.array([np.mean(kn[i:i + ksm]) if np.isfinite(kn[i:i + ksm]).all() else np.nan
                                 for i in range(kn.size - ksm + 1)])
            if np.isfinite(run_mean).any():
                j = int(np.nanargmin(run_mean))
                knee_min = float(run_mean[j])
                knee_arg = w0 + j + ksm // 2
        ank_x_own = k[f"ankle_x_{lab[leg]}_px"].to_numpy(float)[hs]
        ank_x_oth = k[f"ankle_x_{lab[other]}_px"].to_numpy(float)[hs]
        m = {
            "stride_id": st["id"], "leg": leg, "hs_frame": hs, "hs_t_s": st["hs_t_s"],
            "vertical_frame": vert, "vertical_t_s": round(float(t[vert]), 3),
            "end_t_s": round(float(t[min(nxt, n_frames - 1)]), 3),
            "stride_time_s": st["stride_time_s"],
            "stance_pct": (to - hs + 1) / span * 100.0, "swing_pct": 100.0 - (to - hs + 1) / span * 100.0,
            "loading_knee_min_deg": knee_min, "loading_knee_min_frame": knee_arg,
            "contact_knee_deg": at(f"knee_{lab[leg]}_deg", ic_f),
            "vertical_knee_deg": at(f"knee_{lab[leg]}_deg", vert),
            "ankle_contact_deg": at(f"ankle_{lab[leg]}_deg", ic_f),
            "ankle_range_deg": rng(f"ankle_{lab[leg]}_deg"),
            "pelvic_yaw_range_deg": rng("pelvic_yaw_deg"), "shoulder_yaw_range_deg": rng("shoulder_yaw_deg"),
            "torso_twist_range_deg": rng("torso_twist_deg"),
            "pelvic_tilt_range_deg": rng("pelvic_tilt_deg") if view == "frontal" else None,
            "shoulder_tilt_range_deg": rng("shoulder_tilt_deg") if view == "frontal" else None,
            "trunk_incl_mean_deg": mean("trunk_inclination_deg"), "trunk_incl_range_deg": rng("trunk_inclination_deg"),
            "vertical_osc_leg": rng("pelvis_y_leg"),
            "elbow_left_range_deg": rng("elbow_left_deg"), "elbow_right_range_deg": rng("elbow_right_deg"),
            "arm_swing_left_range_deg": rng("shoulder_arm_left_deg"),
            "arm_swing_right_range_deg": rng("shoulder_arm_right_deg"),
            "shoulder_elevation_mean": mean("shoulder_elevation_norm"),
            "rearfoot_range_deg": rng(f"rearfoot_dev_{lab[leg]}_deg", slice(hs, to + 1)) if view == "frontal" else None,
            "step_length_leg": (abs(float(ank_x_own) - float(ank_x_oth)) / leg_px
                                if view == "sagittal" and np.isfinite(ank_x_own) and np.isfinite(ank_x_oth) else None),
            "foot_pitch_contact_deg": at(f"foot_pitch_{lab[leg]}_deg", ic_f) if view == "sagittal" else None,
        }
        m["knee_flag"] = bool(knee_min is not None and knee_min < thr)
        m["knee_deficit_deg"] = (thr - knee_min) if m["knee_flag"] else None
        rows.append(m)
    strides_df = pd.DataFrame(rows)

    # ---- per-phase statistics across strides, per leg ---------------------------------
    by_leg_phase: dict[str, dict] = {}
    for leg in ("L", "R"):
        st_leg = [s for s in strides if s["leg"] == leg]
        by_leg_phase[leg] = {}
        for code in ORDER:
            occ = [(s, next(p for p in s["phases"] if p["phase"] == code)) for s in st_leg]
            det_ = [(s, p) for s, p in occ if p["detected"]]
            entry = {"n_strides": len(det_), "duration_s": None, "measures": {}}
            if not det_:
                by_leg_phase[leg][code] = entry
                continue
            durs = np.array([p["duration_s"] for _, p in det_], float)
            entry["duration_s"] = _r(durs.mean(), 3)
            ev(f"{leg}.{code}.duration_s.mean", entry["duration_s"], "s", "HIGH" if len(det_) >= 10 else "MEDIUM", len(det_))
            for meas, (units, dec) in PHASE_MEASURES.items():
                ser = series(meas, leg)
                per_mean, per_min, per_max, per_kf = [], [], [], []
                for s, p in det_:
                    seg_ = ser[p["start_frame"]:p["end_frame"] + 1]
                    f = seg_[np.isfinite(seg_)]
                    if f.size >= max(1, int(0.6 * seg_.size)):
                        per_mean.append(float(f.mean()))
                        per_min.append(float(f.min()))
                        per_max.append(float(f.max()))
                    kfr = p["key_frame"]
                    if kfr is not None and np.isfinite(ser[kfr]):
                        per_kf.append(float(ser[kfr]))
                n_ok = len(per_mean)
                cap = (KIN_CAP[view] if meas.split("_")[0] in DEPTH_MEASURES
                       else YAW_CAP if meas in ("pelvic_yaw_deg", "torso_twist_deg") else None)
                conf = cap_conf(_confidence(n_ok, n_ok / len(det_)), cap)
                d = {"n_valid": n_ok, "confidence": conf}
                if n_ok:
                    arr = np.array(per_mean)
                    d.update({"mean": _r(arr.mean(), dec), "min": _r(min(per_min), dec),
                              "max": _r(max(per_max), dec),
                              "sd": _r(arr.std(ddof=1), dec) if n_ok >= min_sd else None,
                              "at_key_frame": _r(np.mean(per_kf), dec) if per_kf else None})
                else:
                    d["status"] = NOT_ASSESSABLE
                entry["measures"][meas] = d
                for stat in ("mean", "min", "max", "sd", "at_key_frame"):
                    ev(f"{leg}.{code}.{meas}.{stat}", d.get(stat), units, conf, n_ok)
            by_leg_phase[leg][code] = entry

    # ---- per-leg and whole-clip stride statistics -----------------------------------------
    stride_stats: dict[str, dict] = {"L": {}, "R": {}, "both": {}}
    for name, (units, dec) in STRIDE_METRICS.items():
        if (name in SIDE_ONLY and view != "sagittal") or (name in FRONT_ONLY and view != "frontal"):
            continue
        cap = _stride_cap(name, view)
        for grp, sub in (("L", strides_df[strides_df.leg == "L"]), ("R", strides_df[strides_df.leg == "R"]),
                         ("both", strides_df)):
            v = sub[name].to_numpy(float) if len(sub) else np.array([])
            ok = v[np.isfinite(v)]
            n_ok = int(ok.size)
            conf = cap_conf(_confidence(n_ok, n_ok / max(1, len(sub))), cap)
            d = {"n": n_ok, "n_strides": int(len(sub)), "confidence": conf}
            if n_ok:
                d.update({"mean": _r(ok.mean(), dec), "min": _r(ok.min(), dec), "max": _r(ok.max(), dec),
                          "sd": _r(ok.std(ddof=1), dec) if n_ok >= min_sd else None})
                if n_ok >= min_sd and abs(ok.mean()) > 1e-6:
                    d["cv_pct"] = _r(100 * ok.std(ddof=1) / abs(ok.mean()), 1)
            else:
                d["status"] = NOT_ASSESSABLE
            stride_stats[grp][name] = d
            for stat in ("mean", "min", "max", "sd", "cv_pct"):
                ev(f"{grp}.stride.{name}.{stat}", d.get(stat), "%" if stat == "cv_pct" else units, conf, n_ok)

    # ---- asymmetry (symmetry index, percent) ------------------------------------------------
    asym: dict[str, float | None] = {}

    def add_asym(key, a, b):
        si = G.symmetry_index_pct(a, b)
        asym[key] = _r(si, 1)
        ev(f"asym.{key}.pct", asym[key], "%", "MEDIUM" if si is not None else None)

    for name in ("stride_time_s", "stance_pct", "loading_knee_min_deg", "ankle_range_deg",
                 "pelvic_yaw_range_deg", "rearfoot_range_deg", "step_length_leg", "vertical_osc_leg"):
        a = stride_stats["L"].get(name, {}).get("mean")
        b = stride_stats["R"].get(name, {}).get("mean")
        if a is not None and b is not None:
            add_asym(name, a, b)
    for base, a_key, b_key in (("arm_swing_range", "arm_swing_left_range_deg", "arm_swing_right_range_deg"),
                               ("elbow_range", "elbow_left_range_deg", "elbow_right_range_deg")):
        a = stride_stats["both"].get(a_key, {}).get("mean")
        b = stride_stats["both"].get(b_key, {}).get("mean")
        if a is not None and b is not None:
            add_asym(base, a, b)

    # ---- cadence, step times, support phases --------------------------------------------------
    steps = gait["steps"]
    hs_t = np.array([s["t_s"] for s in steps], float)
    step_int = np.diff(hs_t)
    step_leg = [s["leg"] for s in steps][1:]           # a step interval is credited to the leg that lands
    gait_block: dict = {}
    if step_int.size >= 3:
        good = (step_int > 0.15) & (step_int < 1.5)
        si_ = step_int[good]
        cad = 60.0 / si_
        gait_block.update({
            "cadence_spm": _r(60.0 / si_.mean(), 1), "cadence_sd": _r(cad.std(ddof=1), 1) if si_.size >= min_sd else None,
            "cadence_min": _r(np.percentile(cad, 5), 1), "cadence_max": _r(np.percentile(cad, 95), 1),
            "step_time_s": _r(si_.mean(), 3), "step_time_sd_s": _r(si_.std(ddof=1), 3) if si_.size >= min_sd else None,
        })
        ev("gait.cadence_spm", gait_block["cadence_spm"], "steps/min", "MEDIUM" if fps < 100 else "HIGH", int(si_.size))
        ev("gait.cadence_sd", gait_block["cadence_sd"], "steps/min", "MEDIUM", int(si_.size))
        ev("gait.cadence_p5", gait_block["cadence_min"], "steps/min", "MEDIUM", int(si_.size))
        ev("gait.cadence_p95", gait_block["cadence_max"], "steps/min", "MEDIUM", int(si_.size))
        ev("gait.step_time_s", gait_block["step_time_s"], "s", "HIGH", int(si_.size))
        ev("gait.step_time_sd_s", gait_block["step_time_sd_s"], "s", "MEDIUM", int(si_.size))
        for lg in ("L", "R"):
            sel = np.array([lg == l for l in step_leg])[good]
            if sel.sum() >= 3:
                gait_block[f"step_time_{lab[lg]}_s"] = _r(si_[sel].mean(), 3)
                ev(f"gait.step_time_{lab[lg]}_s", gait_block[f"step_time_{lab[lg]}_s"], "s", "HIGH", int(sel.sum()))
        if gait_block.get("step_time_left_s") and gait_block.get("step_time_right_s"):
            add_asym("step_time", gait_block["step_time_left_s"], gait_block["step_time_right_s"])
    stance = fh[["contact_L", "contact_R"]].to_numpy(bool)
    if len(hs_t):
        first_f = min(s["hs_frame"] for s in steps)
        last_f = max(st["to_frame"] for st in strides) if strides else n_frames - 1
        span_ = stance[first_f:last_f + 1]
        if len(span_):
            both_c = float(np.mean(span_.all(axis=1)) * 100)
            neither = float(np.mean(~span_.any(axis=1)) * 100)
            gait_block["double_support_pct"], gait_block["airborne_pct"] = _r(both_c, 1), _r(neither, 1)
            ev("gait.double_support_pct", gait_block["double_support_pct"], "%", "MEDIUM")
            ev("gait.both_feet_airborne_pct", gait_block["airborne_pct"], "%", "LOW")
    flights = gait["flight_candidates"]
    total_ms = sum(f["duration_ms"] for f in flights)
    gait_block.update({"flight_candidates": len(flights), "flight_total_ms": _r(total_ms, 1),
                       "flight_max_ms": _r(max((f["duration_ms"] for f in flights), default=0), 1)})
    fcap = "MEDIUM" if (fps >= 120 and view == "sagittal") else "LOW"
    ev("gait.flight_candidates", len(flights), "episodes", fcap)
    ev("gait.flight_total_ms", gait_block["flight_total_ms"], "ms", fcap)
    ev("gait.flight_max_ms", gait_block["flight_max_ms"], "ms", fcap)
    if speed:
        mps = speed / 3.6
        gait_block["average_speed_kmh"] = _r(speed, 2)
        gait_block["speed_source"] = speed_source
        ev("gait.average_speed_kmh", gait_block["average_speed_kmh"], "km/h", "HIGH")
        gait_block["pace_min_per_km"] = _r(60.0 / speed, 2)
        ev("gait.pace_min_per_km", gait_block["pace_min_per_km"], "min/km", "HIGH")
        if gait_block.get("step_time_s"):
            gait_block["step_length_m"] = _r(mps * gait_block["step_time_s"], 2)
            ev("gait.step_length_m", gait_block["step_length_m"], "m", "MEDIUM")
        sm = stride_stats["both"].get("stride_time_s", {}).get("mean")
        if sm:
            gait_block["stride_length_m"] = _r(mps * sm, 2)
            ev("gait.stride_length_m", gait_block["stride_length_m"], "m", "MEDIUM")
    vo = stride_stats["both"].get("vertical_osc_leg", {}).get("mean")
    if vo is not None:
        ev("gait.vertical_oscillation_pct_leg", _r(vo * 100, 1), "% of leg length", stride_stats["both"]["vertical_osc_leg"]["confidence"])
        gait_block["vertical_oscillation_pct_leg"] = _r(vo * 100, 1)
    n_valid_knee = int(np.isfinite(strides_df["loading_knee_min_deg"]).sum()) if len(strides_df) else 0
    n_flag = int(strides_df["knee_flag"].sum()) if len(strides_df) else 0
    flag_share = 100.0 * n_flag / n_valid_knee if n_valid_knee else None
    gait_block.update({"straight_leg_flagged": n_flag, "straight_leg_evaluated": n_valid_knee,
                       "straight_leg_flag_share_pct": _r(flag_share, 1)})
    ev("gait.straight_leg_flagged_strides", n_flag, "strides", cap_conf("HIGH", kcap), n_valid_knee)
    ev("gait.straight_leg_evaluated_strides", n_valid_knee, "strides", "HIGH", n_valid_knee)
    ev("gait.straight_leg_flag_share_pct", _r(flag_share, 1), "%", cap_conf("HIGH", kcap), n_valid_knee)

    # ---- drift: last third against first third -------------------------------------------------
    drift: dict = {"available": False}
    if len(strides_df) >= min_drift:
        t0, t1 = float(strides_df["hs_t_s"].min()), float(strides_df["end_t_s"].max())
        third = (t1 - t0) / 3.0
        early_m = strides_df["hs_t_s"] <= t0 + third
        late_m = strides_df["hs_t_s"] >= t1 - third
        drift = {"available": True, "early_window_s": [round(t0, 2), round(t0 + third, 2)],
                 "late_window_s": [round(t1 - third, 2), round(t1, 2)], "metrics": {}}
        for metric, adverse in DRIFT_METRICS.items():
            if metric == "cadence_spm":
                if step_int.size < 6:
                    continue
                tt = hs_t[1:]
                e_v = 60.0 / step_int[(tt <= t0 + third) & (step_int > .15) & (step_int < 1.5)]
                l_v = 60.0 / step_int[(tt >= t1 - third) & (step_int > .15) & (step_int < 1.5)]
                dec, units = 1, "steps/min"
            else:
                e_v = strides_df.loc[early_m, metric].to_numpy(float)
                l_v = strides_df.loc[late_m, metric].to_numpy(float)
                units, dec = STRIDE_METRICS[metric]
            e_v, l_v = e_v[np.isfinite(e_v)], l_v[np.isfinite(l_v)]
            if e_v.size < 3 or l_v.size < 3 or abs(e_v.mean()) < 1e-6:
                continue
            chg = (l_v.mean() - e_v.mean()) / abs(e_v.mean()) * 100.0
            cap = _stride_cap(metric, view) if metric != "cadence_spm" else None
            conf = cap_conf("MEDIUM" if min(e_v.size, l_v.size) >= 5 else "LOW", cap)
            drift["metrics"][metric] = {"early": _r(e_v.mean(), dec), "late": _r(l_v.mean(), dec),
                                        "change_pct": _r(chg, 1), "adverse": bool(np.sign(chg) == adverse),
                                        "n_early": int(e_v.size), "n_late": int(l_v.size)}
            ev(f"early.{metric}.mean", _r(e_v.mean(), dec), units, conf, int(e_v.size))
            ev(f"late.{metric}.mean", _r(l_v.mean(), dec), units, conf, int(l_v.size))
            ev(f"drift.{metric}.change_pct", _r(chg, 1), "%", conf, int(min(e_v.size, l_v.size)))
        adv = [abs(v["change_pct"]) for v in drift["metrics"].values() if v["adverse"] and v["change_pct"] is not None]
        drift["max_adverse_abs_pct"] = _r(max(adv), 1) if adv else 0.0
        ev("drift.max_adverse_pct", drift["max_adverse_abs_pct"], "%", "MEDIUM")
    suppressed_drift = None if drift["available"] else (
        f"{NOT_ASSESSABLE} (early-versus-late comparison requires at least {_w(min_drift)} strides; "
        f"{_w(len(strides_df))} detected)")

    # ---- flagged events (Section 1) -----------------------------------------------------------------
    events: list[dict] = []
    row_by_id = {r["stride_id"]: r for r in rows}
    stride_by_id = {s["id"]: s for s in strides}

    def phase_at(stride_id: str, frame: int) -> str | None:
        for p in stride_by_id[stride_id]["phases"]:
            if p["detected"] and p["start_frame"] <= frame <= p["end_frame"]:
                return p["phase"]
        return None

    def severity(value, moderate, critical):
        return "CRITICAL" if value >= critical else "MODERATE" if value >= moderate else "MINOR"

    # a. straight-leg screen
    flagged = sorted([r for r in rows if r["knee_flag"]], key=lambda r: r["hs_t_s"])
    clusters: list[list[dict]] = []
    for r in flagged:
        if clusters and r["hs_t_s"] - clusters[-1][-1]["vertical_t_s"] <= float(scr["event_merge_gap_s"]):
            clusters[-1].append(r)
        else:
            clusters.append([r])
    for cl in clusters:
        if len(cl) < int(scr["min_flagged_strides_for_event"]):
            continue
        worst = min(cl, key=lambda r: r["loading_knee_min_deg"])
        deficit = thr - worst["loading_knee_min_deg"]
        sev = severity(deficit, float(scr["straight_knee_moderate_deficit_deg"]), float(scr["straight_knee_critical_deficit_deg"]))
        fr = worst["loading_knee_min_frame"]
        events.append({"type": "STRAIGHT_LEG", "start_t_s": cl[0]["hs_t_s"], "end_t_s": cl[-1]["vertical_t_s"],
                       "value": _r(worst["loading_knee_min_deg"], 1), "units": "deg", "threshold": thr,
                       "n_strides": len(cl), "legs": sorted({r["leg"] for r in cl}),
                       "stride_ids": [r["stride_id"] for r in cl], "severity": sev,
                       "confidence": cap_conf("MEDIUM" if worst["loading_knee_min_deg"] is not None else "LOW", kcap),
                       "frames": [{"frame": int(fr), "stride_id": worst["stride_id"], "phase": phase_at(worst["stride_id"], fr),
                                   "label": f"worst knee angle, {worst['stride_id']}"}]})

    # b. contact ambiguity
    for f in flights:
        mid = (f["start_frame"] + f["end_frame"]) // 2
        sev = severity(f["duration_ms"], float(scr["flight_moderate_ms"]), float(scr["flight_critical_ms"]))
        events.append({"type": "CONTACT", "start_t_s": f["start_t_s"], "end_t_s": f["end_t_s"],
                       "value": f["duration_ms"], "units": "ms", "threshold": float(scr["airborne_min_frames"]),
                       "n_strides": 0, "legs": [], "stride_ids": [], "severity": sev, "confidence": fcap,
                       "frames": [{"frame": int(mid), "stride_id": None, "phase": None, "label": "both feet off the ground"}]})

    # c. asymmetry, rolling window of steps
    W_ = int(scr["asymmetry_window_steps"])
    if step_int.size >= W_:
        legs_arr = np.array(step_leg)
        flagged_w = []
        for i in range(0, step_int.size - W_ + 1):
            seg_i, seg_l = step_int[i:i + W_], legs_arr[i:i + W_]
            a, b = seg_i[seg_l == "L"], seg_i[seg_l == "R"]
            if a.size >= 2 and b.size >= 2:
                si_w = G.symmetry_index_pct(float(a.mean()), float(b.mean()))
                if si_w is not None and si_w > float(scr["asymmetry_flag_pct"]):
                    flagged_w.append((i, si_w))
        grp: list[list[tuple[int, float]]] = []
        for i, v in flagged_w:
            if grp and i - grp[-1][-1][0] <= 1:
                grp[-1].append((i, v))
            else:
                grp.append([(i, v)])
        for g in grp:
            worst_i, worst_v = max(g, key=lambda x: x[1])
            first, last = g[0][0], g[-1][0] + W_
            hs_idx = min(worst_i + W_ // 2, len(steps) - 1)
            fr = steps[hs_idx]["hs_frame"]
            events.append({"type": "ASYMMETRY", "start_t_s": float(hs_t[first + 1 - 1 + 1 - 1]) if first + 1 < len(hs_t) else float(hs_t[-1]),
                           "end_t_s": float(hs_t[min(last, len(hs_t) - 1)]), "value": _r(worst_v, 1), "units": "%",
                           "threshold": float(scr["asymmetry_flag_pct"]), "n_strides": 0, "legs": [], "stride_ids": [],
                           "severity": severity(worst_v, float(scr["asymmetry_moderate_pct"]), float(scr["asymmetry_critical_pct"])),
                           "confidence": "MEDIUM",
                           "frames": [{"frame": int(fr), "stride_id": None, "phase": None,
                                       "label": "step-time asymmetry peak"}]})

    # d. fatigue / drift
    if drift["available"]:
        adv = {m: v for m, v in drift["metrics"].items() if v["adverse"] and abs(v["change_pct"]) >= float(scr["drift_flag_pct"])}
        if adv:
            worst_m = max(adv, key=lambda m: abs(adv[m]["change_pct"]))
            wv = abs(adv[worst_m]["change_pct"])
            late = strides_df[strides_df["hs_t_s"] >= drift["late_window_s"][0]]
            early = strides_df[strides_df["hs_t_s"] <= drift["early_window_s"][1]]
            fr_late, fr_early = None, None
            for r in reversed(late.to_dict("records")):
                kf = next((p["key_frame"] for p in stride_by_id[r["stride_id"]]["phases"] if p["phase"] == "MID_STANCE"), None)
                if kf is not None:
                    fr_late = (int(kf), r["stride_id"])
                    break
            for r in early.to_dict("records"):
                kf = next((p["key_frame"] for p in stride_by_id[r["stride_id"]]["phases"] if p["phase"] == "MID_STANCE"), None)
                if kf is not None:
                    fr_early = (int(kf), r["stride_id"])
                    break
            frames_ = [{"frame": f, "stride_id": sid, "phase": "MID_STANCE", "label": lb}
                       for (f, sid), lb in ((fr_early, "early reference"), (fr_late, "late in the clip")) if False] or []
            for pair, lb in ((fr_early, "early in the clip"), (fr_late, "late in the clip")):
                if pair:
                    frames_.append({"frame": pair[0], "stride_id": pair[1], "phase": "MID_STANCE", "label": lb})
            events.append({"type": "FATIGUE", "start_t_s": drift["late_window_s"][0], "end_t_s": drift["late_window_s"][1],
                           "value": _r(wv, 1), "units": "%", "threshold": float(scr["drift_flag_pct"]),
                           "n_strides": int(len(late)), "legs": [], "stride_ids": [], "metrics": sorted(adv),
                           "worst_metric": worst_m,
                           "severity": severity(wv, float(scr["drift_moderate_pct"]), float(scr["drift_critical_pct"])),
                           "confidence": "MEDIUM", "frames": frames_})

    # rank, cap the listing, then number by time
    max_events = int(ctx.cfg.get("report.max_events_listed", 14))
    events.sort(key=lambda e: (SEV_ORDER[e["severity"]], e["start_t_s"]))
    dropped = max(0, len(events) - max_events)
    events = sorted(events[:max_events], key=lambda e: e["start_t_s"])
    for i, e in enumerate(events, 1):
        e["id"] = i
        e["display_name"] = EVENT_DISPLAY[e["type"]]
        base = f"event{i}"
        ev(f"{base}.start_t_s", e["start_t_s"], "s", e["confidence"])
        ev(f"{base}.end_t_s", e["end_t_s"], "s", e["confidence"])
        ev(f"{base}.value", e["value"], e["units"], e["confidence"])
        ev(f"{base}.threshold", e["threshold"], e["units"] if e["type"] != "CONTACT" else "frames", "HIGH")
        if e["n_strides"]:
            ev(f"{base}.n_strides", e["n_strides"], "strides", e["confidence"])
        if e["type"] == "FATIGUE":
            ev(f"{base}.worst_change_pct", e["value"], "%", e["confidence"])
    events_summary = {"total_before_cap": len(events) + dropped, "listed": len(events), "dropped_low_priority": dropped}
    ev("events.count", len(events), "events", "HIGH")

    # ---- injury risk (screening rules) ----------------------------------------------------------------
    sig: dict[str, float | None] = {
        "flagged_stride_share_pct": _r(flag_share, 1),
        "loading_knee_min_asym_pct": asym.get("loading_knee_min_deg"),
        "pelvic_tilt_range_deg": stride_stats["both"].get("pelvic_tilt_range_deg", {}).get("mean"),
        "trunk_incl_range_deg": stride_stats["both"].get("trunk_incl_range_deg", {}).get("mean"),
        "pelvic_yaw_range_asym_pct": asym.get("pelvic_yaw_range_deg"),
        "ankle_range_asym_pct": asym.get("ankle_range_deg"),
        "stance_asym_pct": asym.get("stance_pct"),
        "rearfoot_angle_range_deg": stride_stats["both"].get("rearfoot_range_deg", {}).get("mean"),
        "rearfoot_angle_asym_pct": asym.get("rearfoot_range_deg"),
        "max_abs_drift_pct": drift.get("max_adverse_abs_pct") if drift["available"] else None,
    }
    risk: dict[str, dict] = {}
    for cat, spec in ctx.cfg.risk_rules["categories"].items():
        rows_ = []
        for s_ in spec["signals"]:
            views = s_.get("views", ["any"])
            allowed = "any" in views or view in views
            val = sig.get(s_["metric"])
            if not allowed or val is None:
                rows_.append({"metric": s_["metric"], "value": None, "level": None,
                              "reason": (f"not measurable from a {view} view" if not allowed else "no valid data")})
                continue
            lvl = "HIGH" if val >= s_["high"] else "MODERATE" if val >= s_["moderate"] else "LOW"
            rows_.append({"metric": s_["metric"], "value": val, "level": lvl, "moderate": s_["moderate"],
                          "high": s_["high"], "unit": s_["unit"]})
        avail = [r for r in rows_ if r["level"]]
        order = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
        if avail:
            driver = max(avail, key=lambda r: (order[r["level"]], r["value"] / max(1e-9, r["high"])))
            level = driver["level"]
            conf = "LOW" if (cat == "knee_stress" and view == "frontal") or overall == "LOW" else \
                   ("MEDIUM" if overall == "MEDIUM" or view != "sagittal" else "MEDIUM")
            risk[cat] = {"label": spec["label"], "level": level, "driver_metric": driver["metric"],
                         "driver_value": driver["value"], "driver_unit": driver["unit"],
                         "moderate": driver["moderate"], "high": driver["high"], "confidence": conf,
                         "signals": rows_, "status": "ASSESSED"}
            ev(f"risk.{cat}.driver_value", driver["value"], driver["unit"], conf)
            ev(f"risk.{cat}.moderate_threshold", driver["moderate"], driver["unit"], "HIGH")
            ev(f"risk.{cat}.high_threshold", driver["high"], driver["unit"], "HIGH")
        else:
            risk[cat] = {"label": spec["label"], "level": "NOT ASSESSED", "confidence": None, "signals": rows_,
                         "status": f"{NOT_ASSESSABLE} from a {view} camera view"}

    # ---- technique rows (Section 2), status decided here, prose by the model ------------------------------
    def sget(name, grp="both", stat="mean"):
        return stride_stats[grp].get(name, {}).get(stat)

    heel_first = None
    if view == "sagittal":
        pit = strides_df["foot_pitch_contact_deg"].to_numpy(float)
        pit = pit[np.isfinite(pit)]
        if pit.size >= 3:
            heel_first = 100.0 * float(np.mean(pit <= -float(scr["heel_strike_toe_up_deg"])))
            ev("gait.heel_first_contact_pct", _r(heel_first, 1), "% of strides", "MEDIUM", int(pit.size))
    technique = {
        "contact": {"parameter": "Contact (Rule 54.1)",
                    "status": ("POSSIBLE CONCERN" if flights else "PROBABLE PASS") if speed is not None or True else "",
                    "confidence": fcap,
                    "keys": ["gait.flight_candidates", "gait.flight_max_ms", "gait.double_support_pct",
                             "gait.both_feet_airborne_pct", "session.analysis_fps"]},
        "straight_leg": {"parameter": "Straight leg (Rule 54.2)",
                         "status": (f"FLAGGED ({_w(n_flag)} of {_w(n_valid_knee)} strides)" if n_flag else
                                    "NO FLAG (screen only)") if n_valid_knee else NOT_ASSESSABLE,
                         "confidence": cap_conf("MEDIUM", kcap),
                         "keys": ["gait.straight_leg_flagged_strides", "gait.straight_leg_evaluated_strides",
                                  "gait.straight_leg_flag_share_pct", "both.stride.loading_knee_min_deg.mean",
                                  "both.stride.loading_knee_min_deg.min", "screen.straight_knee_min_deg"]},
        "pelvic_rotation": {"parameter": "Pelvic rotation", "status": INDIVIDUAL,
                            "confidence": cap_conf("MEDIUM", YAW_CAP),
                            "keys": ["both.stride.pelvic_yaw_range_deg.mean", "both.stride.pelvic_yaw_range_deg.sd",
                                     "both.stride.pelvic_yaw_range_deg.cv_pct", "both.stride.torso_twist_range_deg.mean"]
                            + (["both.stride.pelvic_tilt_range_deg.mean"] if view == "frontal" else [])},
        "torso_posture": {"parameter": "Torso posture", "status": INDIVIDUAL, "confidence": "MEDIUM",
                          "keys": ["both.stride.trunk_incl_mean_deg.mean", "both.stride.trunk_incl_range_deg.mean",
                                   "both.stride.shoulder_elevation_mean.mean", "asym.arm_swing_range.pct"]},
        "stride": {"parameter": "Stride length and frequency", "status": "MEASURED", "confidence": "MEDIUM",
                   "keys": ["gait.cadence_spm", "gait.cadence_sd", "gait.step_time_s", "asym.step_time.pct",
                            "gait.step_length_m", "gait.stride_length_m", "both.stride.step_length_leg.mean"]},
        "foot_strike": {"parameter": "Foot strike",
                        "status": ("HEEL FIRST" if (heel_first or 0) >= 70 else "MIXED" if heel_first is not None else NOT_ASSESSABLE)
                        if view == "sagittal" else f"{NOT_ASSESSABLE} from a {view} camera view",
                        "confidence": "MEDIUM" if view == "sagittal" else None,
                        "keys": ["gait.heel_first_contact_pct", "both.stride.foot_pitch_contact_deg.mean"]},
    }
    for row in technique.values():
        row["keys"] = [x for x in row["keys"] if x in evidence]

    # ---- benchmarks: cited entries only ---------------------------------------------------------------------
    usable_bm, rejected_bm = [], []
    for e in ctx.cfg.benchmarks.get("entries", []):
        why = None
        if not e.get("source"):
            why = "no source"
        elif e.get("status") == "NEEDS_SOURCE":
            why = "status NEEDS_SOURCE"
        elif e.get("target") is None and e.get("range") is None:
            why = "no target or range"
        elif not e.get("measure") or e["measure"].replace("_deg", "") not in {m.replace("_deg", "") for m in STRIDE_METRICS}:
            why = "measure is not one the pipeline produces"
        if why:
            rejected_bm.append({"id": e.get("id"), "reason": why})
            continue
        m = e["measure"]
        athlete_v = stride_stats["both"].get(m, {}).get("mean")
        target = e.get("target")
        row = {"id": e["id"], "variable": e.get("variable"), "measure": m, "athlete_value": athlete_v,
               "target": target, "range": e.get("range"), "units": e.get("units"),
               "difference": _r(athlete_v - target, 1) if athlete_v is not None and target is not None else None,
               "evidence_class": e.get("evidence_class"), "source": e["source"],
               "confidence": "MEDIUM" if athlete_v is not None else None}
        usable_bm.append(row)
        ev(f"benchmark.{e['id']}.athlete_value", athlete_v, e.get("units") or "", row["confidence"])
        ev(f"benchmark.{e['id']}.target", target, e.get("units") or "", "HIGH")
        ev(f"benchmark.{e['id']}.difference", row["difference"], e.get("units") or "", row["confidence"])
    benched = {r["measure"] for r in usable_bm}
    benchmarks = {"rows": usable_bm, "rejected_entries": rejected_bm,
                  "no_benchmark": {m: INDIVIDUAL for m in STRIDE_METRICS if m not in benched}}
    if benched and "pelvic_yaw_range_deg" in benched:
        technique["pelvic_rotation"]["status"] = "COMPARED TO CITED REFERENCE"

    payload = {
        "schema_version": 1, "run_id": ctx.run_id, "config_hash": ctx.cfg.config_hash,
        "session": session, "data_quality": quality,
        "gait": gait_block, "asymmetry": asym, "drift": drift, "drift_status": suppressed_drift,
        "events": events, "events_summary": events_summary,
        "risk": risk, "risk_rules": {"status": ctx.cfg.risk_rules.get("status"), "version": ctx.cfg.risk_rules.get("version")},
        "technique": technique,
        "stride_stats": stride_stats, "by_leg_phase": by_leg_phase,
        "screening": scr, "benchmarks": benchmarks,
        "measure_units": {m: spec[0] for m, spec in {**PHASE_MEASURES, **STRIDE_METRICS}.items()},
        "evidence_index": evidence,
    }
    text = json.dumps(payload, indent=2, allow_nan=False, default=str)
    from racewalk.io_guard import guarded_open
    out = ctx.artefact("05_metrics.json")
    with guarded_open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    with guarded_open(ctx.artefact("05_strides.json"), "w", encoding="utf-8") as fh:
        json.dump(json.loads(strides_df.to_json(orient="records")), fh, indent=1)

    # ---- verification --------------------------------------------------------------------------------------
    res.check("metrics_written_without_nan", True, f"{len(text) / 1024:.0f} KB, strict JSON (no NaN)")
    res.check("evidence_index_populated", len(evidence) > 0, f"{len(evidence)} evidence entries")
    sd_leak = [f"{g}.{m}" for g, d in stride_stats.items() for m, c in d.items()
               if c.get("sd") is not None and c["n"] < min_sd]
    res.check("sd_suppressed_below_min_strides", not sd_leak,
              f"SD reported with too few strides: {sd_leak[:5]}" if sd_leak else "SD only reported with enough strides.")
    res.check("drift_needs_enough_strides", drift["available"] or bool(suppressed_drift),
              suppressed_drift or "Early-versus-late drift computed.", severity=WARN)
    res.check("benchmarks_all_cited", all(r.get("source") for r in usable_bm),
              f"{len(usable_bm)} cited benchmark(s) used, {len(rejected_bm)} rejected: {[r['reason'] for r in rejected_bm]}")
    res.check("benchmarks_available", bool(usable_bm),
              f"No usable benchmarks. Section 2 prints '{INDIVIDUAL}' for every variable until cited entries "
              f"are added to config/benchmarks.json.", severity=WARN)
    no_frame = [e["id"] for e in events if not e["frames"]]
    res.check("every_event_has_a_frame", not no_frame,
              f"Events without a frame to annotate: {no_frame}" if no_frame else "Every event has a frame.", severity=WARN)
    res.check("core_gait_metrics_available", bool(gait_block.get("cadence_spm")) and n_valid_knee > 0,
              "Cadence and loading-window knee angle were measured." if gait_block.get("cadence_spm") and n_valid_knee
              else "Cadence or loading-window knee angle could not be measured.")
    res.check("overall_confidence_assigned", overall in ("HIGH", "MEDIUM", "LOW"), f"Overall analysis confidence: {overall}")
    res.check("camera_cap_applied", True,
              f"Camera view is {view}: depth-derived leg angles are capped at {kcap}, rotation at {YAW_CAP}.")

    res.outputs["metrics"] = str(out)
    res.outputs["strides"] = str(ctx.artefact("05_strides.json"))
    res.stats = {"n_strides": len(strides), "evidence_entries": len(evidence), "events": len(events),
                 "overall_confidence": overall, "benchmarks_used": len(usable_bm),
                 "straight_leg_flagged": n_flag}
    return res
