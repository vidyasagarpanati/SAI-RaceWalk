"""Synthetic race walker generator.

Builds a physically plausible 33-landmark sequence with KNOWN geometry so S3, S4 and S5
can be exercised without a video, and so a wrong answer is provable. Used by the tests
and by ``racewalk selftest``.

Model (metres, y grows downward, the athlete walks toward -z, away from a rear camera):
  * two-segment legs (0.44 m thigh, 0.44 m shank) swinging about the hip
  * per-leg phase u in [0,1): stance u in [0, 0.55), swing u in [0.55, 1)
    the leg sweeps from +A to -A over u in [0, 0.5], holds to 0.55 (10 percent double
    support), then swings forward with knee flexion
  * stance knee straight, except on chosen strides where it is bent by ``bend_deg``
  * pelvis height is set so the loaded foot is exactly on the ground, so vertical
    oscillation and foot lift come out of the geometry, not out of a fudge
  * pelvis and shoulder yaw, opposite arm swing, optional fatigue drift and a brief
    both-feet-airborne episode

The camera is an orthographic projection: ``rear`` shows (x, y), ``side`` shows (-z, y).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from racewalk import geometry as G
from racewalk.landmarks import ID, side

FPS = 60.0
FRAME_W, FRAME_H = 640, 360
L1 = L2 = 0.44          # thigh, shank
HIP_HALF, SH_HALF = 0.09, 0.19
S_Y = 0.509             # frame heights per metre
STANCE_END, STANCE_SWEEP = 0.55, 0.50


def _smooth_bump(s):
    s = np.clip(s, 0.0, 1.0)
    return np.sin(np.pi * s)


def build(duration_s: float = 30.0, stride_s: float = 0.76, view: str = "rear", seed: int = 5,
          noise_px: float = 0.5, A_deg: float = 22.0, swing_flex_deg: float = 42.0,
          bent: dict | None = None, bend_deg: float = 25.0, pelvic_yaw_amp: float = 10.0,
          drift: bool = False, flight_at_s: float | None = None,
          asym_R_stride_scale: float = 1.0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = int(FPS * duration_s)
    t = np.arange(n) / FPS
    A = np.radians(A_deg)
    bent = bent or {}

    # phase accumulator (stride frequency slows late in the clip when ``drift``)
    f0 = 1.0 / stride_s
    freq = np.full(n, f0)
    if drift:
        ramp = np.clip((t - duration_s * 2 / 3) / (duration_s / 3), 0, 1)
        freq = f0 * (1.0 - 0.14 * ramp)
    phase = np.cumsum(freq) / FPS
    uL = phase % 1.0
    strideL = np.floor(phase).astype(int)
    uR = (phase + 0.5) % 1.0
    strideR = np.floor(phase + 0.5).astype(int) - 1     # first R stride index 0 ends up at phase 0.5

    def leg_angles(u, stride, leg):
        alpha = np.where(u < STANCE_END,
                         A - 2 * A * np.minimum(u / STANCE_SWEEP, 1.0),
                         -A + 2 * A * (u - STANCE_END) / (1 - STANCE_END))
        s = np.clip((u - STANCE_END) / (1 - STANCE_END), 0, 1)
        # the swing knee flexes, then is fully extended for the last quarter of swing:
        # a race walker reaches for the ground with a straight leg
        beta = np.where(u >= STANCE_END, np.radians(swing_flex_deg) * _smooth_bump(s / 0.72), 0.0)
        b = np.zeros(n)
        for k, deg in bent.get(leg, {}).items():
            m = (stride == k) & (u < 0.30)
            b[m] = np.radians(deg) * (1 - np.clip(u[m] / 0.30, 0, 1) ** 2)
        beta = beta + b
        return alpha, beta

    aL, bL = leg_angles(uL, strideL, "L")
    aR, bR = leg_angles(uR, strideR, "R")

    def depth(alpha, beta):
        return L1 * np.cos(alpha) + L2 * np.cos(alpha - beta)

    dL, dR = depth(aL, bL), depth(aR, bR)
    support_is_L = uL < 0.5
    d_support = np.where(support_is_L, dL, dR)
    ypel = 0.88 - d_support
    if flight_at_s is not None:
        k = int(flight_at_s * FPS)
        win = np.zeros(n)
        win[k:k + 5] = np.array([0.4, 1.0, 1.0, 1.0, 0.4]) * 0.035
        ypel = ypel - win

    yaw_amp = np.full(n, np.radians(pelvic_yaw_amp))
    if drift:
        ramp = np.clip((t - duration_s * 2 / 3) / (duration_s / 3), 0, 1)
        yaw_amp = yaw_amp * (1.0 - 0.40 * ramp)
    pel_yaw = yaw_amp * np.cos(2 * np.pi * uL)
    sh_yaw = -0.7 * pel_yaw

    W = {}     # name -> (n, 3) absolute coordinates (x, y, z) in metres

    def put(name, x, y, z):
        W[name] = np.stack([x, y, z], axis=1)

    zero = np.zeros(n)
    put("MID", zero, ypel, zero)
    # hips
    for sgn, nm in ((-1, "LEFT"), (1, "RIGHT")):
        put(f"{nm}_HIP", sgn * HIP_HALF * np.cos(pel_yaw), ypel, -sgn * HIP_HALF * np.sin(pel_yaw) * -1)
    # legs
    for nm, alpha, beta, sgn in (("LEFT", aL, bL, -1), ("RIGHT", aR, bR, 1)):
        hip = W[f"{nm}_HIP"]
        knee = hip + np.stack([np.zeros(n), L1 * np.cos(alpha), -L1 * np.sin(alpha)], axis=1)
        ank = knee + np.stack([np.zeros(n), L2 * np.cos(alpha - beta), -L2 * np.sin(alpha - beta)], axis=1)
        W[f"{nm}_KNEE"], W[f"{nm}_ANKLE"] = knee, ank
        W[f"{nm}_HEEL"] = ank + np.array([0.0, 0.07, 0.05])
        W[f"{nm}_FOOT_INDEX"] = ank + np.stack([np.zeros(n), np.full(n, 0.07) - 0.02 * np.sin(alpha - beta), np.full(n, -0.16)], axis=1)
    # trunk
    for sgn, nm in ((-1, "LEFT"), (1, "RIGHT")):
        put(f"{nm}_SHOULDER", sgn * SH_HALF * np.cos(sh_yaw), ypel - 0.50, -sgn * SH_HALF * np.sin(sh_yaw) * -1)
    # arms swing opposite to the leg on the same side
    for nm, alpha, sgn in (("LEFT", aR, -1), ("RIGHT", aL, 1)):
        g = 0.85 * alpha
        sh = W[f"{nm}_SHOULDER"]
        el = sh + np.stack([np.full(n, sgn * 0.05), 0.30 * np.cos(g), -0.30 * np.sin(g)], axis=1)
        wr = el + np.stack([np.full(n, -sgn * 0.02), -0.27 * np.sin(g), -0.27 * np.cos(g)], axis=1)
        W[f"{nm}_ELBOW"], W[f"{nm}_WRIST"] = el, wr
        d = wr - el
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        W[f"{nm}_INDEX"] = wr + 0.05 * d
        W[f"{nm}_PINKY"] = wr + 0.04 * d + np.array([0.01, 0.0, 0.0])
        W[f"{nm}_THUMB"] = wr + 0.04 * d - np.array([0.01, 0.0, 0.0])
    # head
    head_y = ypel - 0.50
    put("NOSE", zero, head_y - 0.20, zero - 0.09)
    for nm, dx in (("LEFT_EYE_INNER", -0.015), ("LEFT_EYE", -0.03), ("LEFT_EYE_OUTER", -0.045),
                   ("RIGHT_EYE_INNER", 0.015), ("RIGHT_EYE", 0.03), ("RIGHT_EYE_OUTER", 0.045)):
        put(nm, zero + dx, head_y - 0.23, zero - 0.08)
    put("LEFT_EAR", zero - 0.08, head_y - 0.21, zero)
    put("RIGHT_EAR", zero + 0.08, head_y - 0.21, zero)
    put("MOUTH_LEFT", zero - 0.02, head_y - 0.15, zero - 0.08)
    put("MOUTH_RIGHT", zero + 0.02, head_y - 0.15, zero - 0.08)

    pts3 = np.zeros((n, 33, 3))
    for name, arr in W.items():
        if name in ID:
            pts3[:, ID[name]] = arr

    # Ground-truth joint angles, computed directly from the noiseless 3D geometry above,
    # using the exact same vertex/proximal/distal landmark triples as s03_kinematics.py.
    # This is what the pipeline's measured angles are validated against.
    true_angles: dict[str, np.ndarray] = {}
    for lg, prefix in (("left", "LEFT"), ("right", "RIGHT")):
        q = side(prefix)
        true_angles[f"knee_{lg}_deg"] = G.angle_at(pts3[:, q["knee"]], pts3[:, q["hip"]], pts3[:, q["ankle"]])
        true_angles[f"hip_{lg}_deg"] = G.angle_at(pts3[:, q["hip"]], pts3[:, q["shoulder"]], pts3[:, q["knee"]])
        true_angles[f"ankle_{lg}_deg"] = G.angle_at(pts3[:, q["ankle"]], pts3[:, q["knee"]], pts3[:, q["foot"]])
        true_angles[f"elbow_{lg}_deg"] = G.angle_at(pts3[:, q["elbow"]], pts3[:, q["shoulder"]], pts3[:, q["wrist"]])

    # projection to normalised image coordinates
    sx = S_Y * FRAME_H / FRAME_W
    if view == "rear":
        xn = 0.5 + pts3[:, :, 0] * sx
    else:
        xn = 0.5 + (-pts3[:, :, 2]) * sx
    yn = 0.446 + (pts3[:, :, 1] - 0.0) * S_Y
    px_noise = noise_px / np.array([FRAME_W, FRAME_H])
    img = np.stack([xn, yn], axis=2) + rng.normal(0, 1, (n, 33, 2)) * px_noise

    # world landmarks: hip-centred, metres
    pelvis = (pts3[:, ID["LEFT_HIP"]] + pts3[:, ID["RIGHT_HIP"]]) / 2.0
    world = pts3 - pelvis[:, None, :]
    world[:, :, :2] += rng.normal(0, 0.004, (n, 33, 2))
    world[:, :, 2] += rng.normal(0, 0.012, (n, 33))

    vis = np.full((n, 33), 0.93)
    vis[:, 0:11] = 0.62 if view == "rear" else 0.9        # face barely visible from behind
    frame_idx = np.repeat(np.arange(n, dtype=np.int32), 33)
    df = pd.DataFrame({
        "frame": frame_idx,
        "t_ms": (frame_idx * (1000.0 / FPS)).astype(np.float32),
        "lm": np.tile(np.arange(33, dtype=np.int16), n),
        "x": img[:, :, 0].reshape(-1), "y": img[:, :, 1].reshape(-1), "z": np.zeros(n * 33),
        "visibility": vis.reshape(-1), "presence": vis.reshape(-1),
        "wx": world[:, :, 0].reshape(-1), "wy": world[:, :, 1].reshape(-1),
        "wz": world[:, :, 2].reshape(-1),
    })

    def crossings(u, offset_units):
        """Times where the phase counter passes k + offset."""
        ph = phase + offset_units
        k = np.floor(ph).astype(int)
        idx = np.where(np.diff(k) > 0)[0] + 1
        return [float(t[i]) for i in idx]

    truth = {
        "fps": FPS, "n_frames": n, "view": view, "frame": (FRAME_W, FRAME_H),
        "true_angles": true_angles,
        "hs_L_s": crossings(uL, 0.0),
        "hs_R_s": crossings(uR, 0.5),
        "to_L_s": crossings(uL, 1 - STANCE_END),
        "to_R_s": crossings(uR, 0.5 + 1 - STANCE_END),
        "bent": bent, "bend_deg": bend_deg, "flight_at_s": flight_at_s,
        "stride_s": stride_s, "pelvic_yaw_amp_deg": pelvic_yaw_amp, "drift": drift,
    }
    return df, truth
