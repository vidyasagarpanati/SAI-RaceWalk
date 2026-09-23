"""S4 gait.

IN     03_kinematics.parquet, 03_quality.json, 01_frames.json,
       config/phase_rules.yaml, session manual_event_overrides
DO     deterministic foot-contact detection, then heel strikes, toe-offs, strides,
       the vertical (mid-stance) event and the six stride phases
OUT    04_gait.json, 04_foot_height.parquet
VERIFY both legs produce strides, heel strikes alternate, phases never overlap, every
       detected phase has a usable key frame, identical input -> identical output

How contact is found (all thresholds in phase_rules.yaml):
  1. FOOT HEIGHT  lowest foot point (heel or toe) above that foot's own ground level,
                  in leg lengths. Ground = a high rolling percentile of the foot's
                  lowest image position, so a treadmill belt, a track and a slightly
                  tilted camera all work without calibration.
  2. CONTACT      hysteresis on foot height, thresholds set as fractions of that
                  foot's own swing amplitude, so they scale with the athlete and the
                  camera. Runs shorter than min_contact_s are noise.
  3. EVENTS       heel strike = first frame of a contact run, toe-off = last frame.
  4. STRIDE       one leg's heel strike to its next heel strike.
  5. VERTICAL     the moment the opposite foot is at its highest inside this leg's
                  contact. It needs no side view, which is why it works from behind.
  6. PHASES       INITIAL_CONTACT, LOADING, MID_STANCE, PUSH_OFF, EARLY_SWING,
                  LATE_SWING, contiguous and in that order.

A phase the signals cannot support is written as detected=false with the reason. It is
never filled in with a guessed boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from racewalk import geometry as G
from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.phase_defs import BADGE, DISPLAY, ORDER


# ---------------------------------------------------------------- helpers
def _frames(seconds: float, fps: float, minimum: int = 1) -> int:
    return max(minimum, int(round(seconds * fps)))


def contact_mask(h: np.ndarray, enter: float, exit_: float, max_gap: int) -> np.ndarray:
    """Hysteresis state machine over foot height. True while the foot is on the ground."""
    out = np.zeros(h.size, dtype=bool)
    state, gap = False, 0
    for i, v in enumerate(h):
        if not np.isfinite(v):
            if state:
                gap += 1
                if gap > max_gap:
                    state = False
            out[i] = state
            continue
        gap = 0
        if not state and v < enter:
            state = True
        elif state and v > exit_:
            state = False
        out[i] = state
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, mask.size - 1))
    return out


def clean_runs(rs: list[tuple[int, int]], min_contact: int, min_swing: int) -> list[tuple[int, int]]:
    """Merge contact runs separated by a too-short swing, then drop too-short contacts."""
    merged: list[list[int]] = []
    for s, e in rs:
        if merged and s - merged[-1][1] - 1 < min_swing:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s + 1 >= min_contact]


def grade(signal: np.ndarray, idx: int, threshold: float | None, usable: np.ndarray, h: int) -> str:
    """HIGH clean crossing, MEDIUM oscillating, LOW missing data around the boundary."""
    lo, hi = max(0, idx - h), min(signal.size, idx + h + 1)
    window = signal[lo:hi]
    if window.size == 0 or not np.all(np.isfinite(window)) or not np.all(usable[lo:hi]):
        return "LOW"
    if threshold is None:
        return "HIGH"
    crossings = int(np.sum(np.diff(np.sign(window - threshold)) != 0))
    return "HIGH" if crossings <= 1 else "MEDIUM"


def key_frame(start: int, end: int, n_visible: np.ndarray, usable: np.ndarray,
              prefer: int | None = None) -> int | None:
    """Best-measured frame in a phase: the preferred event frame if usable, otherwise
    the frame with the most visible landmarks nearest the midpoint."""
    if end < start:
        return None
    idx = np.arange(start, end + 1)
    ok = idx[usable[start:end + 1]]
    if ok.size == 0:
        return None
    if prefer is not None and start <= prefer <= end and usable[prefer]:
        return int(prefer)
    mid = (start + end) / 2.0
    return int(max(ok, key=lambda i: (n_visible[i], -abs(i - mid))))


def foot_height(low_y: np.ndarray, leg_px: float, win: int, pct: float) -> tuple[np.ndarray, np.ndarray]:
    ground = G.rolling_percentile(low_y, win, pct)
    h = (ground - low_y) / leg_px
    return np.where(np.isfinite(h), np.clip(h, 0.0, None), np.nan), ground


# ---------------------------------------------------------------- main
def run(ctx: Context) -> StepResult:
    res = StepResult(step="S4")
    rules = ctx.cfg.phase_rules
    c, ph, bc = rules["contact"], rules["phases"], rules["boundary_confidence"]

    fps = float(ctx.read_json("01_frames.json")["analysis_fps"])
    k = pd.read_parquet(ctx.artefact("03_kinematics.parquet"))
    n = len(k)
    t = k["t_s"].to_numpy()
    usable = k["frame_usable"].to_numpy().astype(bool)
    n_visible = k["n_visible_landmarks"].to_numpy()
    leg_px = float(k["leg_length_px"].iloc[0])
    h_ = int(bc["hysteresis_frames"])

    win = _frames(c["ground_window_s"], fps, 5) | 1
    heights, grounds = {}, {}
    for leg, label in (("L", "left"), ("R", "right")):
        heights[leg], grounds[leg] = foot_height(k[f"foot_low_y_{label}_px"].to_numpy(), leg_px,
                                                 win, float(c["ground_percentile"]))

    legs_out, contacts, states, raw_states = {}, {}, {}, {}
    overrides = ctx.session.get("manual_event_overrides") or {}
    for leg in ("L", "R"):
        hgt = heights[leg]
        fin = hgt[np.isfinite(hgt)]
        amp = float(np.percentile(fin, 95)) if fin.size else 0.0
        enter, exit_ = c["enter_frac_of_amplitude"] * amp, c["exit_frac_of_amplitude"] * amp
        usable_amp = amp >= float(c["min_amplitude_leg_lengths"])
        mask = contact_mask(hgt, enter, exit_, _frames(c["max_nan_gap_s"], fps)) if usable_amp else np.zeros(n, bool)
        rs = clean_runs(runs(mask), _frames(c["min_contact_s"], fps), _frames(c["min_swing_s"], fps))
        source = "rules"
        if overrides.get(leg):
            rs = []
            for pair in overrides[leg]:
                a0 = int(np.searchsorted(t, float(pair[0])))
                a1 = int(np.searchsorted(t, float(pair[1]), side="right")) - 1
                if a1 > a0:
                    rs.append((a0, a1))
            source = "manual"
        contacts[leg] = rs
        raw_states[leg] = mask
        st = np.zeros(n, bool)
        for s, e in rs:
            st[s:e + 1] = True
        states[leg] = st
        legs_out[leg] = {"amplitude_leg_lengths": round(amp, 4), "enter_leg_lengths": round(enter, 4),
                         "exit_leg_lengths": round(exit_, 4), "amplitude_usable": usable_amp,
                         "n_contacts": len(rs), "source": source,
                         "contacts": [[int(s), int(e)] for s, e in rs]}

    # ---- strides and phases ------------------------------------------------
    min_stride, max_stride = _frames(c["min_stride_s"], fps), _frames(c["max_stride_s"], fps)
    ic_n = _frames(ph["initial_contact_s"], fps)
    mh = _frames(ph["mid_stance_half_window_s"], fps)
    strides, skipped = [], []
    for leg, other in (("L", "R"), ("R", "L")):
        rs = contacts[leg]
        for i, (hs, to) in enumerate(rs):
            if i + 1 >= len(rs):
                skipped.append({"leg": leg, "hs_frame": int(hs), "reason": "no following heel strike (clip ends)"})
                continue
            nxt = rs[i + 1][0]
            if not (min_stride <= nxt - hs <= max_stride):
                skipped.append({"leg": leg, "hs_frame": int(hs),
                                "reason": f"stride of {(nxt - hs) / fps:.2f} s is outside "
                                          f"[{c['min_stride_s']}, {c['max_stride_s']}] s"})
                continue
            hc = heights[other][hs:to + 1]
            vert, method = (hs + to) // 2, "midpoint (opposite foot never rose)"
            if np.isfinite(hc).any():
                j = int(np.nanargmax(hc))
                if 0 < j < len(hc) - 1 and hc[j] >= 0.5 * float(np.nanmax(heights[other])) * 0.2:
                    vert, method = hs + j, "opposite foot highest"
            own = heights[leg][to:nxt]
            peak = to + int(np.nanargmax(own)) if own.size and np.isfinite(own).any() else (to + nxt) // 2

            b = {
                "INITIAL_CONTACT": (hs, min(hs + ic_n - 1, vert - mh - 1 if vert - mh - 1 >= hs else hs + ic_n - 1)),
                "LOADING": (hs + ic_n, vert - mh - 1),
                "MID_STANCE": (vert - mh, vert + mh),
                "PUSH_OFF": (vert + mh + 1, to),
                "EARLY_SWING": (to + 1, peak),
                "LATE_SWING": (peak + 1, nxt - 1),
            }
            prefer = {"INITIAL_CONTACT": hs, "MID_STANCE": vert, "PUSH_OFF": to, "LATE_SWING": nxt - 1}
            thr = {"INITIAL_CONTACT": c["enter_frac_of_amplitude"] * legs_out[leg]["amplitude_leg_lengths"],
                   "PUSH_OFF": c["exit_frac_of_amplitude"] * legs_out[leg]["amplitude_leg_lengths"]}
            phases, last_end = [], hs - 1
            for code in ORDER:
                s0, e0 = b[code]
                start, end = max(int(s0), last_end + 1), int(e0)
                detected = end >= start
                kf = key_frame(start, end, n_visible, usable, prefer.get(code)) if detected else None
                conf = None
                if detected:
                    edge = {"INITIAL_CONTACT": hs, "PUSH_OFF": to}.get(code, start)
                    conf = grade(heights[leg], edge, thr.get(code), usable, h_)
                phases.append({
                    "phase": code, "badge": BADGE[code], "display_name": DISPLAY[code],
                    "detected": detected,
                    "start_frame": start if detected else None, "end_frame": end if detected else None,
                    "start_t_s": round(float(t[start]), 3) if detected else None,
                    "end_t_s": round(float(t[end]), 3) if detected else None,
                    "duration_s": round((end - start + 1) / fps, 3) if detected else None,
                    "key_frame": kf,
                    "key_frame_t_s": round(float(t[kf]), 3) if kf is not None else None,
                    "boundary_confidence": conf,
                    "reason_not_detected": None if detected else "phase is shorter than one frame at this rate",
                })
                if detected:
                    last_end = end
            strides.append({
                "id": f"{leg}{sum(1 for s in strides if s['leg'] == leg) + 1:02d}", "leg": leg,
                "index": sum(1 for s in strides if s["leg"] == leg),
                "hs_frame": int(hs), "to_frame": int(to), "next_hs_frame": int(nxt),
                "vertical_frame": int(vert), "vertical_method": method,
                "hs_t_s": round(float(t[hs]), 3), "to_t_s": round(float(t[to]), 3),
                "stride_time_s": round((nxt - hs) / fps, 3), "phases": phases,
            })
    strides.sort(key=lambda s: s["hs_frame"])

    # ---- steps: every heel strike, both legs, in time order ---------------------
    hs_all = sorted([(hs, leg) for leg in ("L", "R") for hs, _ in contacts[leg]])
    steps = [{"leg": lg, "hs_frame": int(f), "t_s": round(float(t[f]), 3)} for f, lg in hs_all]
    alternates = sum(1 for a, b in zip(hs_all, hs_all[1:]) if a[1] != b[1])
    alt_share = alternates / max(1, len(hs_all) - 1)

    # ---- both-feet-airborne candidates ---------------------------------------
    # Raw hysteresis states, NOT the cleaned runs: cleaning merges any swing gap shorter
    # than min_swing_s into the contact, which would hide exactly the short episodes
    # this screen exists to find.
    air = ~(raw_states["L"] | raw_states["R"])
    valid = np.isfinite(heights["L"]) & np.isfinite(heights["R"]) & usable
    min_air = int(ctx.cfg.phase_rules["screening"]["airborne_min_frames"])
    first_hs = min((r[0][0] for r in contacts.values() if r), default=0)
    last_to = max((r[-1][1] for r in contacts.values() if r), default=n - 1)
    flights = []
    for s, e in runs(air & valid):
        if e - s + 1 >= min_air and first_hs <= s and e <= last_to:
            flights.append({"start_frame": int(s), "end_frame": int(e),
                            "start_t_s": round(float(t[s]), 3), "end_t_s": round(float(t[e]), 3),
                            "duration_ms": round((e - s + 1) / fps * 1000.0, 1),
                            "peak_height_leg_lengths": round(float(max(np.nanmax(heights["L"][s:e + 1]),
                                                                        np.nanmax(heights["R"][s:e + 1]))), 4)})

    def pct(x):
        x = x[np.isfinite(x)]
        return ({f"p{q}": round(float(np.percentile(x, q)), 4) for q in (1, 5, 25, 50, 75, 95, 99)}
                if x.size else None)

    payload = {
        "analysis_fps": fps, "rules_version": rules.get("version"), "thresholds_used": rules,
        "leg_length_px": round(leg_px, 1), "n_strides": len(strides),
        "n_strides_by_leg": {lg: sum(1 for s in strides if s["leg"] == lg) for lg in ("L", "R")},
        "legs": legs_out, "strides": strides, "steps": steps,
        "heel_strike_alternation_share": round(alt_share, 4),
        "flight_candidates": flights, "skipped_strides": skipped,
        "signal_summary": {"foot_height_left_leg_lengths": pct(heights["L"]),
                           "foot_height_right_leg_lengths": pct(heights["R"]),
                           "frames_left_in_contact": int(states["L"].sum()),
                           "frames_right_in_contact": int(states["R"].sum())},
    }
    out = ctx.write_json("04_gait.json", payload)
    from racewalk.io_guard import guarded_path
    fh_path = guarded_path(ctx.artefact("04_foot_height.parquet"))
    pd.DataFrame({"frame": np.arange(n), "t_s": t,
                  "height_L": heights["L"], "height_R": heights["R"],
                  "ground_L_px": grounds["L"], "ground_R_px": grounds["R"],
                  "contact_L": states["L"], "contact_R": states["R"]}).to_parquet(fh_path, index=False)

    # ---- verification ---------------------------------------------------------
    for lg in ("L", "R"):
        m = legs_out[lg]
        res.check(f"leg_{lg}_foot_signal_usable", m["amplitude_usable"],
                  f"Swing amplitude {m['amplitude_leg_lengths']} leg lengths "
                  f"(needs {c['min_amplitude_leg_lengths']}). If too low the foot landmarks are not "
                  f"tracking, or the foot never leaves the ground in the image. Read signal_summary.")
    n_l, n_r = payload["n_strides_by_leg"]["L"], payload["n_strides_by_leg"]["R"]
    res.check("strides_detected", n_l >= 3 and n_r >= 3,
              f"{n_l} left and {n_r} right strides detected." if n_l >= 3 and n_r >= 3 else
              f"Only {n_l} left and {n_r} right strides. At least three per leg are needed. "
              f"Read signal_summary in 04_gait.json and adjust config/phase_rules.yaml.")
    res.check("heel_strikes_alternate", alt_share >= 0.90,
              f"{alt_share:.0%} of consecutive heel strikes alternate between legs. A low value means "
              f"one leg's contact detection is merging or splitting steps.", severity=WARN)
    overlap, gaps, missing_kf, undetected = [], [], [], []
    for st in strides:
        prev = None
        for p in st["phases"]:
            if not p["detected"]:
                undetected.append(f"{st['id']} {p['phase']}")
                continue
            if p["key_frame"] is None:
                missing_kf.append(f"{st['id']} {p['phase']}")
            if prev is not None:
                if p["start_frame"] <= prev["end_frame"]:
                    overlap.append(f"{st['id']} {prev['phase']}/{p['phase']}")
                elif p["start_frame"] > prev["end_frame"] + 1:
                    gaps.append(f"{st['id']} {prev['phase']}->{p['phase']}")
            prev = p
    res.check("phases_do_not_overlap", not overlap, f"Overlaps: {overlap[:6]}" if overlap else "No overlaps.")
    res.check("phases_contiguous", not gaps, f"Gaps: {gaps[:6]}" if gaps else "No gaps.", severity=WARN)
    res.check("key_frames_usable", len(missing_kf) <= 0.1 * max(1, 6 * len(strides)),
              f"No usable key frame for {len(missing_kf)} phase(s), e.g. {missing_kf[:5]}" if missing_kf else
              "Every detected phase has a key frame with adequate landmark quality.", severity=WARN)
    res.check("phases_all_detected", len(undetected) <= 0.1 * max(1, 6 * len(strides)),
              f"{len(undetected)} phase(s) not resolvable at this frame rate, e.g. {undetected[:5]}"
              if undetected else "All six phases detected in every stride.", severity=WARN)
    res.check("skipped_strides_few", len(skipped) <= 0.15 * max(1, len(strides) + len(skipped)),
              f"{len(skipped)} contact(s) could not form a valid stride.", severity=WARN)

    res.outputs["gait"] = str(out)
    res.outputs["foot_height"] = str(fh_path)
    res.stats = {"n_strides": len(strides), "left": n_l, "right": n_r,
                 "alternation_share": round(alt_share, 3), "flight_candidates": len(flights),
                 "undetected_phases": len(undetected)}
    return res
