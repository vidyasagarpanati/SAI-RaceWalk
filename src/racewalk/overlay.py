"""Shared annotation renderer for S6 (key frames) and S7 (every video frame).

One renderer, so the annotated screenshots in the report and the annotated video can never
disagree. It draws the skeleton, the reference lines, live joint angles, the phase of
BOTH legs (each leg is in some phase at every instant), foot-contact markers, any active
flag, and a side panel with the measurement table, an observation and a coaching box.
It returns an account of what it drew, so S6 verifies coverage mechanically instead of
trusting that it looks right.

Layout: the video frame is kept intact on the left and a side panel is added on the
right. Text never covers the athlete. Joint-angle labels sit on the frame with leader
lines, placed by a collision check that keeps them off every landmark.

Numbers drawn here are passed in by the caller. S6 passes the rounded values from
05_metrics.json or 03_kinematics.parquet, so the image shows what the evidence file holds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from racewalk import landmarks as L
from racewalk.phase_defs import BADGE, COLOR_BGR, DISPLAY

# BGR colours
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
GREEN = (60, 200, 60)
ORANGE = (0, 150, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 230, 255)
RED = (50, 50, 235)
BLUE = (235, 130, 40)
GRAY = (175, 175, 175)
BG = (25, 25, 25)
LEFT_COL, RIGHT_COL = ORANGE, BLUE

FONT = cv2.FONT_HERSHEY_SIMPLEX

REQUIRED_KEY_FRAME_ELEMENTS = [
    "landmarks", "landmark_ids", "topology", "origin_pelvis", "gravity_line", "pelvis_line",
    "shoulder_line", "trunk_line", "leg_lines", "foot_contact_markers", "joint_angle_labels",
    "phase_badge", "phase_name", "timestamp", "frame_number", "stride_id",
    "measurement_table", "observation_callout", "coaching_box", "flag_panel",
]


@dataclass
class RenderReport:
    drawn: set[str] = field(default_factory=set)
    skipped: dict[str, str] = field(default_factory=dict)
    n_labels: int = 0
    label_joint_overlaps: int = 0

    def did(self, name: str) -> None:
        self.drawn.add(name)
        self.skipped.pop(name, None)

    def skip(self, name: str, why: str) -> None:
        if name not in self.drawn:
            self.skipped[name] = why

    def missing(self, required: list[str]) -> list[str]:
        return [r for r in required if r not in self.drawn and r not in self.skipped]


# ------------------------------------------------------------------ primitives
def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _ok(p) -> bool:
    return p is not None and np.all(np.isfinite(p))


def dashed(img, p1, p2, color, thick=2, dash=14, gap=9, dotted=False):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1:
        return
    d = (p2 - p1) / length
    step = dash + gap
    pos = 0.0
    while pos < length:
        a = p1 + d * pos
        if dotted:
            cv2.circle(img, _pt(a), max(1, thick), color, -1, cv2.LINE_AA)
        else:
            b = p1 + d * min(pos + dash, length)
            cv2.line(img, _pt(a), _pt(b), color, thick, cv2.LINE_AA)
        pos += step


def extended(p1, p2, w: int, h: int):
    """Endpoints of the infinite line through p1,p2, clipped to the image."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d /= n
    far = (w + h) * 2
    a, b = p1 - d * far, p1 + d * far
    ok, q1, q2 = cv2.clipLine((0, 0, w, h), _pt(a), _pt(b))
    return (q1, q2) if ok else None


def star(img, c, r, color):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((c[0] + rad * math.cos(ang), c[1] + rad * math.sin(ang)))
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def diamond(img, c, r, color):
    pts = np.array([(c[0], c[1] - r), (c[0] + r, c[1]), (c[0], c[1] + r), (c[0] - r, c[1])], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def wrap(text: str, width_chars: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------ label layer
class Labels:
    """Collects text labels, places them without collisions, then draws them
    with one semi-transparent background pass."""

    def __init__(self, w: int, h: int, s: float, joints: list[tuple[int, int]]):
        self.w, self.h, self.s = w, h, s
        self.joints = joints
        self.rects: list[tuple[int, int, int, int]] = []
        self.ops: list[dict] = []
        self.overlaps = 0

    def _size(self, text, scale, thick):
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        return tw, th + base

    def _free(self, r, avoid_joints=True):
        x0, y0, x1, y1 = r
        if x0 < 0 or y0 < 0 or x1 > self.w or y1 > self.h:
            return False
        for a in self.rects:
            if not (x1 < a[0] or x0 > a[2] or y1 < a[1] or y0 > a[3]):
                return False
        if avoid_joints:
            m = int(6 * self.s)
            for jx, jy in self.joints:
                if x0 - m <= jx <= x1 + m and y0 - m <= jy <= y1 + m:
                    return False
        return True

    def add(self, anchor, text, color, scale=0.55, thick=1, leader=True, fixed=None):
        scale *= self.s
        thick = max(1, int(round(thick * self.s)))
        tw, th = self._size(text, scale, thick)
        pad = int(4 * self.s)
        bw, bh = tw + 2 * pad, th + 2 * pad
        if fixed is not None:
            rect = (fixed[0], fixed[1], fixed[0] + bw, fixed[1] + bh)
        else:
            ax, ay = _pt(anchor)
            rect = None
            for radius in (38, 64, 96, 132):
                r = radius * self.s
                for ang in (-30, 30, -150, 150, -90, 90, 0, 180):
                    cx = ax + r * math.cos(math.radians(ang))
                    cy = ay + r * math.sin(math.radians(ang))
                    x0 = int(cx - bw / 2 if abs(math.cos(math.radians(ang))) < 0.3
                             else (cx if math.cos(math.radians(ang)) > 0 else cx - bw))
                    y0 = int(cy - bh / 2)
                    cand = (x0, y0, x0 + bw, y0 + bh)
                    if self._free(cand):
                        rect = cand
                        break
                if rect:
                    break
            if rect is None:   # fall back: nearest in-bounds spot, counted as an overlap
                x0 = int(min(max(0, ax + 20 * self.s), self.w - bw))
                y0 = int(min(max(0, ay - bh - 10 * self.s), self.h - bh))
                rect = (x0, y0, x0 + bw, y0 + bh)
                if not self._free(rect):
                    self.overlaps += 1
        self.rects.append(rect)
        self.ops.append({"rect": rect, "text": text, "color": color, "scale": scale,
                         "thick": thick, "pad": pad, "th": th,
                         "anchor": _pt(anchor) if (leader and anchor is not None) else None})

    def draw(self, img, alpha=0.62):
        layer = img.copy()
        for op in self.ops:
            x0, y0, x1, y1 = op["rect"]
            cv2.rectangle(layer, (x0, y0), (x1, y1), BG, -1)
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)
        for op in self.ops:
            x0, y0, x1, y1 = op["rect"]
            if op["anchor"] is not None:
                ax, ay = op["anchor"]
                ex = min(max(ax, x0), x1)
                ey = min(max(ay, y0), y1)
                cv2.line(img, (ax, ay), (ex, ey), op["color"], max(1, op["thick"]), cv2.LINE_AA)
            cv2.putText(img, op["text"], (x0 + op["pad"], y1 - op["pad"] - int(0.25 * op["th"])),
                        FONT, op["scale"], op["color"], op["thick"], cv2.LINE_AA)


def _fmt(v, dec=1, unit=" deg"):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{dec}f}{unit}"



def _fmt(v, dec=1, unit=" deg"):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{dec}f}{unit}"


# ------------------------------------------------------------------ main entry
def render(frame: np.ndarray, pts: np.ndarray, vis: np.ndarray, values: dict, *,
           view_class: str, legs: dict, stride_label: str | None, t_s: float, frame_idx: int,
           mode: str = "key", track_conf: float = 0.5,
           measurements: list[tuple] | None = None,
           observation: str | None = None, coaching: str | None = None,
           extras: dict | None = None, event: dict | None = None,
           focus_leg: str | None = None, screen_knee_deg: float | None = None,
           flags: list[str] | None = None,
           near_side: str | None = None) -> tuple[np.ndarray, RenderReport]:
    """Annotate one frame.

    pts      (33, 2) pixel coordinates, NaN where undetected
    vis      (33,) landmark confidence
    values   measure name -> number (angles in degrees)
    legs     {"L": {"phase": code|None, "contact": bool|None}, "R": {...}}
    event    {"id": int, "display": str, "severity": str} when the frame illustrates a flag
    near_side  in a side view, the leg/arm facing the camera ("L" or "R"). The far side's
               joint-angle labels and table rows are dimmed, not removed: the value is still
               measured, just not visually confirmable from this camera angle.
    mode     "key" draws everything including IDs and the full panel;
             "video" draws a lighter set tuned for motion
    """
    extras = extras or {}
    rep = RenderReport()
    h, w = frame.shape[:2]
    s = h / 1080.0
    img = frame.copy()
    good = np.isfinite(pts).all(axis=1) & (np.nan_to_num(vis, nan=0) >= track_conf)
    present = np.isfinite(pts).all(axis=1)
    ID = L.ID

    def mid(i, j):
        return (pts[i] + pts[j]) / 2.0 if present[i] and present[j] else None

    pelvis = mid(ID["LEFT_HIP"], ID["RIGHT_HIP"])
    sg = mid(ID["LEFT_SHOULDER"], ID["RIGHT_SHOULDER"])
    lw = max(1, int(round(2 * s)))
    colr = {"L": LEFT_COL, "R": RIGHT_COL}

    # ---- reference lines, underneath the skeleton ----------------------------------
    if _ok(pelvis):
        dashed(img, (pelvis[0], 0), (pelvis[0], h), MAGENTA, lw)
        rep.did("gravity_line")
    else:
        rep.skip("gravity_line", "hip landmarks not detected")
    if present[ID["LEFT_HIP"]] and present[ID["RIGHT_HIP"]]:
        seg = extended(pts[ID["LEFT_HIP"]], pts[ID["RIGHT_HIP"]], w, h)
        if seg:
            cv2.line(img, seg[0], seg[1], GREEN, lw, cv2.LINE_AA)
            rep.did("pelvis_line")
    else:
        rep.skip("pelvis_line", "hip landmarks not detected")
    if present[ID["LEFT_SHOULDER"]] and present[ID["RIGHT_SHOULDER"]]:
        seg = extended(pts[ID["LEFT_SHOULDER"]], pts[ID["RIGHT_SHOULDER"]], w, h)
        if seg:
            cv2.line(img, seg[0], seg[1], ORANGE, lw, cv2.LINE_AA)
            rep.did("shoulder_line")
    else:
        rep.skip("shoulder_line", "shoulder landmarks not detected")
    if _ok(sg) and _ok(pelvis):
        dashed(img, sg, pelvis, YELLOW, lw)
        rep.did("trunk_line")
    else:
        rep.skip("trunk_line", "shoulder or hip landmarks not detected")

    # ---- skeleton --------------------------------------------------------------------
    for a, b in L.CONNECTIONS:
        if present[a] and present[b]:
            col = L.REGION_COLOR_BGR[L.REGION_OF[a]]
            if not (good[a] and good[b]):
                col = tuple(int(c * 0.45) for c in col)
            cv2.line(img, _pt(pts[a]), _pt(pts[b]), col, lw, cv2.LINE_AA)
    rep.did("topology")

    # legs emphasised over the skeleton, one colour per leg
    n_leg = 0
    for lg, prefix in (("L", "LEFT"), ("R", "RIGHT")):
        q = L.side(prefix)
        if all(present[q[j]] for j in ("hip", "knee", "ankle")):
            thick = lw + (2 if focus_leg == lg else 1)
            cv2.polylines(img, [np.array([_pt(pts[q[j]]) for j in ("hip", "knee", "ankle")])], False,
                          colr[lg], thick, cv2.LINE_AA)
            n_leg += 1
    if n_leg:
        rep.did("leg_lines")
    else:
        rep.skip("leg_lines", "no complete leg detected")

    r_pt = max(2, int(round(4 * s)))
    id_scale = 0.38 * s
    for i in range(33):
        if not present[i]:
            continue
        col = L.REGION_COLOR_BGR[L.REGION_OF[i]]
        if good[i]:
            cv2.circle(img, _pt(pts[i]), r_pt, col, -1, cv2.LINE_AA)
            cv2.circle(img, _pt(pts[i]), r_pt, WHITE, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, _pt(pts[i]), max(1, r_pt - 2), tuple(int(c * 0.45) for c in col), -1, cv2.LINE_AA)
        if mode == "key" and i >= 11:
            txt = str(i) if good[i] else f"{i} ({float(np.nan_to_num(vis[i])):.2f})"
            cv2.putText(img, txt, (_pt(pts[i])[0] + r_pt + 1, _pt(pts[i])[1] - r_pt), FONT, id_scale,
                        col if good[i] else GRAY, 1, cv2.LINE_AA)
    rep.did("landmarks")
    if mode == "key":
        rep.did("landmark_ids")

    # ---- foot contact markers ---------------------------------------------------------
    joints_px = [_pt(pts[i]) for i in range(33) if present[i]]
    labels = Labels(w, h, s, joints_px)
    n_contact = 0
    for lg, prefix in (("L", "LEFT"), ("R", "RIGHT")):
        heel = L.ID[f"{prefix}_HEEL"]
        c = (legs.get(lg) or {}).get("contact")
        if present[heel] and c is not None:
            x, y = _pt(pts[heel])
            d = int(9 * s)
            if c:
                cv2.rectangle(img, (x - d, y - d), (x + d, y + d), colr[lg], -1)
            else:
                cv2.rectangle(img, (x - d, y - d), (x + d, y + d), colr[lg], max(1, lw))
            labels.add(pts[heel], f"{lg} {'CONTACT' if c else 'AIR'}", colr[lg], 0.42)
            n_contact += 1
    if n_contact:
        rep.did("foot_contact_markers")
    else:
        rep.skip("foot_contact_markers", "heel landmarks not detected or contact state unknown")

    # ---- origin and reference-line labels -----------------------------------------------
    if _ok(pelvis):
        cv2.circle(img, _pt(pelvis), int(9 * s), GREEN, max(2, lw), cv2.LINE_AA)
        labels.add(pelvis, "PELVIS ORIGIN", GREEN, 0.45)
        rep.did("origin_pelvis")
    else:
        rep.skip("origin_pelvis", "hip landmarks not detected")
    if mode == "key":
        if _ok(pelvis):
            labels.add(None, "GRAVITY / VERTICAL", MAGENTA, 0.42, leader=False,
                       fixed=(int(pelvis[0] + 6 * s), int(8 * s)))
        if "pelvis_line" in rep.drawn and view_class == "frontal":
            labels.add(pts[ID["RIGHT_HIP"]], f"PELVIC TILT {_fmt(values.get('pelvic_tilt_deg'))}", GREEN, 0.45)
        if "shoulder_line" in rep.drawn and view_class == "frontal":
            labels.add(pts[ID["RIGHT_SHOULDER"]], f"SHOULDER TILT {_fmt(values.get('shoulder_tilt_deg'))}", ORANGE, 0.45)
        if "trunk_line" in rep.drawn:
            tl = "TRUNK LEAN" if view_class == "sagittal" else "TRUNK TILT"
            labels.add((sg + pelvis) / 2, f"{tl} {_fmt(values.get('trunk_inclination_deg'))}", YELLOW, 0.45)

    # ---- joint-angle labels -----------------------------------------------------------------
    n_joint = 0
    for lg, prefix, label in (("L", "LEFT", "left"), ("R", "RIGHT", "right")):
        q = L.side(prefix)
        ph = (legs.get(lg) or {}).get("phase")
        far = view_class == "sagittal" and near_side is not None and lg != near_side
        for name, jk, key in (("KNEE", "knee", f"knee_{label}_deg"), ("HIP", "hip", f"hip_{label}_deg"),
                              ("ANKLE", "ankle", f"ankle_{label}_deg")):
            if not present[q[jk]]:
                continue
            v = values.get(key)
            col = colr[lg]
            if far:
                col = tuple(int(c * 0.4) for c in col)
            elif (name == "KNEE" and screen_knee_deg is not None and v is not None
                    and ph in ("INITIAL_CONTACT", "LOADING", "MID_STANCE") and v < screen_knee_deg):
                col = RED
            labels.add(pts[q[jk]], f"{lg} {name}: {_fmt(v)}", col)
            n_joint += 1
        if present[q["elbow"]]:
            col = tuple(int(c * 0.4) for c in colr[lg]) if far else colr[lg]
            labels.add(pts[q["elbow"]], f"{lg} ELBOW: {_fmt(values.get(f'elbow_{label}_deg'))}", col, 0.5)
            n_joint += 1
    if n_joint:
        rep.did("joint_angle_labels")
    else:
        rep.skip("joint_angle_labels", "no joint landmarks detected")

    # ---- phase-specific annotations ----------------------------------------------------------------
    if mode == "key" and focus_leg:
        _leg_specific(img, labels, pts, present, values, legs, focus_leg, screen_knee_deg, extras, rep, s, lw)

    labels.draw(img)
    rep.n_labels = len(labels.ops)
    rep.label_joint_overlaps = labels.overlaps

    # ---- badges, event banner, timestamp -------------------------------------------------------------
    _badges(img, legs, stride_label, t_s, frame_idx, s, rep)
    _flag_banner(img, event, flags, s, rep)

    canvas = _panel(img, mode, legs, stride_label, t_s, frame_idx, measurements or [], observation,
                    coaching, rep, s, event, focus_leg, near_side, view_class)
    return canvas, rep


def _leg_specific(img, labels, pts, present, values, legs, focus_leg, screen, extras, rep, s, lw):
    """Extras for the leg the frame is about: the knee gauge and the foot-height bar."""
    prefix = "LEFT" if focus_leg == "L" else "RIGHT"
    q = L.side(prefix)
    ph = (legs.get(focus_leg) or {}).get("phase")
    label = "left" if focus_leg == "L" else "right"
    if present[q["knee"]] and present[q["hip"]] and present[q["ankle"]] and ph:
        v = values.get(f"knee_{label}_deg")
        bad = screen is not None and v is not None and v < screen and ph in ("INITIAL_CONTACT", "LOADING", "MID_STANCE")
        if ph in ("INITIAL_CONTACT", "LOADING", "MID_STANCE") and screen is not None:
            txt = (f"STRAIGHT-LEG SCREEN: {focus_leg} knee {_fmt(v)} is BELOW the screen of {screen:.0f} deg"
                   if bad else f"STRAIGHT-LEG SCREEN: {focus_leg} knee {_fmt(v)}, at or above {screen:.0f} deg")
            labels.add(pts[q["knee"]], txt, RED if bad else GREEN, 0.5)
            # extend the thigh line straight through the knee: the shank should follow it
            d = pts[q["knee"]] - pts[q["hip"]]
            n = np.linalg.norm(d)
            if n > 1:
                end = pts[q["knee"]] + d / n * np.linalg.norm(pts[q["ankle"]] - pts[q["knee"]])
                dashed(img, pts[q["knee"]], end, WHITE, lw)
    hgt = extras.get("foot_height_norm", {}).get(focus_leg)
    if hgt is not None and present[q["heel"]]:
        x, y = _pt(pts[q["heel"]])
        labels.add(pts[q["heel"]], f"FOOT HEIGHT {hgt:.3f} LL", colr_leg(focus_leg), 0.42)
    trail = np.asarray(extras.get("pelvis_trail", []), float)
    if trail.size:
        trail = trail[np.isfinite(trail).all(axis=1)]
        if len(trail) >= 3:
            cv2.polylines(img, [trail.astype(np.int32)], False, CYAN, lw, cv2.LINE_AA)
    rep.did("leg_specific")


def colr_leg(lg):
    return LEFT_COL if lg == "L" else RIGHT_COL


def _badges(img, legs, stride_label, t_s, frame_idx, s, rep):
    x0, y0 = int(14 * s), int(14 * s)
    bw, bh = int(360 * s), int(46 * s)
    any_phase = False
    for n_, lg in enumerate(("L", "R")):
        ph = (legs.get(lg) or {}).get("phase")
        col = COLOR_BGR.get(ph, (90, 90, 90)) if ph else (90, 90, 90)
        y = y0 + n_ * (bh + int(6 * s))
        layer = img.copy()
        cv2.rectangle(layer, (x0, y), (x0 + bw, y + bh), col, -1)
        cv2.addWeighted(layer, 0.78, img, 0.22, 0, dst=img)
        cv2.rectangle(img, (x0, y), (x0 + bw, y + bh), WHITE, max(1, int(s)))
        txt = f"{lg}  PHASE {BADGE[ph]}  {DISPLAY[ph].upper()}" if ph else f"{lg}  (no stride phase)"
        cv2.putText(img, txt[:44], (x0 + int(10 * s), y + int(31 * s)), FONT, 0.58 * s, WHITE,
                    max(1, int(2 * s)), cv2.LINE_AA)
        any_phase = any_phase or bool(ph)
    if any_phase:
        rep.did("phase_badge")
        rep.did("phase_name")
    else:
        rep.skip("phase_badge", "frame is outside every detected stride")
        rep.skip("phase_name", "frame is outside every detected stride")
    yy = y0 + 2 * (bh + int(6 * s)) + int(24 * s)
    info = f"STRIDE {stride_label or '-'}   t = {t_s:.3f} s   FRAME {frame_idx}"
    cv2.putText(img, info, (x0, yy), FONT, 0.6 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    rep.did("timestamp")
    rep.did("frame_number")
    rep.did("stride_id")


def _flag_banner(img, event, flags, s, rep):
    h, w = img.shape[:2]
    lines = []
    if event:
        lines.append((f"EVENT {event['id']}: {event['display'].upper()} [{event['severity']}]", RED))
    for f in flags or []:
        lines.append((f, ORANGE))
    if not lines:
        rep.did("flag_panel")
        return
    y = h - int(16 * s) - (len(lines) - 1) * int(34 * s)
    for text, col in lines:
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.66 * s, max(1, int(2 * s)))
        x = int((w - tw) / 2)
        layer = img.copy()
        cv2.rectangle(layer, (x - 10, y - th - 10), (x + tw + 10, y + 10), BG, -1)
        cv2.addWeighted(layer, 0.75, img, 0.25, 0, dst=img)
        cv2.putText(img, text, (x, y), FONT, 0.66 * s, col, max(1, int(2 * s)), cv2.LINE_AA)
        y += int(34 * s)
    rep.did("flag_panel")


def _panel(img, mode, legs, stride_label, t_s, frame_idx, measurements, observation, coaching, rep, s,
           event, focus_leg, near_side=None, view_class=None):
    h, w = img.shape[:2]
    pw = max(int(0.46 * h), 380)
    panel = np.full((h, pw, 3), 22, np.uint8)
    x, y = int(18 * s), int(40 * s)
    lh = int(30 * s)
    sc = 0.6 * s
    th = max(1, int(round(1.5 * s)))
    chars = max(24, int(pw / (11.5 * s)))

    def line(text, color=WHITE, scale=sc, bold=False):
        nonlocal y
        cv2.putText(panel, text, (x, y), FONT, scale, color, th + (1 if bold else 0), cv2.LINE_AA)
        y += lh

    ph = (legs.get(focus_leg) or {}).get("phase") if focus_leg else None
    title = f"STRIDE {stride_label or '-'}" + (f"  |  {DISPLAY[ph].upper()}" if ph else "")
    bar_col = COLOR_BGR.get(ph, (70, 70, 70)) if ph else (70, 70, 70)
    cv2.rectangle(panel, (0, 0), (pw, int(56 * s)), bar_col, -1)
    y = int(38 * s)
    line(title[:34], WHITE, 0.62 * s, True)
    y += int(10 * s)
    line(f"t = {t_s:.3f} s   frame {frame_idx}", GRAY)
    y += int(8 * s)
    line("MEASUREMENT SUMMARY", YELLOW, sc, True)
    for row in measurements:
        label, value = row[0], row[1]
        dim = bool(row[2]) if len(row) > 2 else False
        lab_col = tuple(int(c * 0.55) for c in GRAY) if dim else GRAY
        val_col = tuple(int(c * 0.5) for c in WHITE) if dim else WHITE
        cv2.putText(panel, label, (x, y), FONT, 0.52 * s, lab_col, th, cv2.LINE_AA)
        (vw, _), _ = cv2.getTextSize(value, FONT, 0.52 * s, th)
        cv2.putText(panel, value, (pw - x - vw, y), FONT, 0.52 * s, val_col, th, cv2.LINE_AA)
        y += int(26 * s)
    if measurements:
        rep.did("measurement_table")
    else:
        rep.skip("measurement_table", "no measurements supplied")
    if mode == "key":
        for title_, text, name, col in (("OBSERVATION", observation, "observation_callout", CYAN),
                                        ("COACHING IMPLICATION", coaching, "coaching_box", ORANGE)):
            y += int(12 * s)
            line(title_, col, sc, True)
            if text:
                for ln in wrap(text, chars)[:9]:
                    line(ln, WHITE, 0.5 * s)
                rep.did(name)
            else:
                line("(not supplied)", GRAY, 0.5 * s)
                rep.skip(name, "no text supplied")
    y = h - int(118 * s)
    line("LEGEND", GRAY, 0.5 * s, True)
    legend_lines = [("left leg (orange)   right leg (blue)", WHITE), ("gravity . pelvis . shoulders . trunk", MAGENTA),
                    ("filled square = foot in contact", WHITE)]
    if view_class == "sagittal" and near_side is not None:
        legend_lines.append((f"dim = far side ({'right' if near_side == 'L' else 'left'}), not clearly visible from this angle", GRAY))
    for txt, col in legend_lines:
        line(txt, col, 0.45 * s)
    canvas = np.hstack([img, panel])
    ch, cw = canvas.shape[:2]
    if cw % 2 or ch % 2:   # x264 needs even dimensions
        canvas = cv2.copyMakeBorder(canvas, 0, ch % 2, 0, cw % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas
