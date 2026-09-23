"""Angle and signal maths. Pure NumPy, no model, no randomness."""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def savgol_coeffs(window: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    """Savitzky-Golay filter coefficients, built with NumPy so SciPy is not a dependency."""
    if window % 2 == 0:
        window += 1
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    from math import factorial
    return pinv[deriv] * factorial(deriv)


def smooth_nan(series: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing that preserves NaN gaps.

    Gaps are bridged by linear interpolation for the convolution, then re-masked,
    so a smoothed value is never invented where there was no measurement.
    """
    y = np.asarray(series, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < max(window, polyorder + 2):
        return y
    idx = np.arange(y.size)
    filled = np.interp(idx, idx[mask], y[mask])
    coeffs = savgol_coeffs(window, polyorder)
    half = len(coeffs) // 2
    padded = np.pad(filled, half, mode="edge")
    out = np.convolve(padded, coeffs[::-1], mode="valid")
    out[~mask] = np.nan
    return out


def angle_at(b: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle in degrees at vertex ``b`` formed by a-b-c.

    Arrays are (n, d). Works in 2D or 3D. Returns NaN where any point is NaN.
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.einsum("ij,ij->i", v1, v2) / (n1 * n2 + EPS)
    cos = np.clip(cos, -1.0, 1.0)
    out = np.degrees(np.arccos(cos))
    bad = ~np.isfinite(n1) | ~np.isfinite(n2) | (n1 < EPS) | (n2 < EPS)
    out[bad] = np.nan
    return out


def line_angle_from_horizontal(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Signed angle of the line p->q from the image horizontal, degrees.

    Positive means q sits lower on screen than p (y grows downward), which for a
    left-to-right landmark pair reads as a downward tilt toward q.
    """
    d = q - p
    return np.degrees(np.arctan2(d[:, 1], d[:, 0] + EPS))


def line_angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Inclination of a segment from the image vertical (gravity), degrees.

    0 means perfectly plumb. Positive means the top point is to the right of the
    bottom point.
    """
    d = top - bottom
    return np.degrees(np.arctan2(d[:, 0], -(d[:, 1]) + EPS))


def signed_tilt(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Tilt of the p-q line relative to horizontal, folded to [-90, 90]."""
    a = line_angle_from_horizontal(p, q)
    a = (a + 180.0) % 180.0
    return np.where(a > 90.0, a - 180.0, a)


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def speed(points: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference speed of a point series. Units are input units per second."""
    n = points.shape[0]
    out = np.full(n, np.nan)
    if n < 3 or dt <= 0:
        return out
    d = np.linalg.norm(points[2:] - points[:-2], axis=-1)
    out[1:-1] = d / (2.0 * dt)
    return out


def rate_of_change(series: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference derivative of a 1D series, units per second."""
    y = np.asarray(series, dtype=float)
    out = np.full(y.size, np.nan)
    if y.size < 3 or dt <= 0:
        return out
    out[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    return out


def rolling_std(series: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling standard deviation, NaN-tolerant."""
    y = np.asarray(series, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    half = max(1, window // 2)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = y[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2:
            out[i] = float(np.std(seg, ddof=1))
    return out


# ---------------------------------------------------------------- gait helpers
def yaw_from_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Rotation of the left-to-right line about the vertical axis, degrees.

    Inputs are MediaPipe WORLD landmarks (n, 3): x to the athlete's left in the camera
    frame, z toward the camera. 0 means the line is parallel to the image plane. Depth
    is the least reliable output of monocular pose estimation, so callers treat this as
    an inferred quantity and report its range, never an absolute orientation.
    """
    d = right - left
    return np.degrees(np.arctan2(d[:, 2], d[:, 0] + EPS))


def unwrap_centered(yaw_deg: np.ndarray) -> np.ndarray:
    """Unwrap a yaw series and centre it on its own median, NaN-safe.

    The absolute yaw depends on how the camera sits, the excursion around the athlete's
    own centre does not.
    """
    y = np.asarray(yaw_deg, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return y
    idx = np.arange(y.size)
    filled = np.interp(idx, idx[mask], y[mask])
    unwrapped = np.degrees(np.unwrap(np.radians(filled)))
    out = unwrapped - np.nanmedian(unwrapped[mask])
    out[~mask] = np.nan
    return out


def rolling_percentile(series: np.ndarray, window: int, q: float) -> np.ndarray:
    """Centred rolling percentile, NaN-tolerant. Used for the per-foot ground level."""
    y = np.asarray(series, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    half = max(1, window // 2)
    for i in range(n):
        seg = y[max(0, i - half): min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.percentile(seg, q))
    return out


def symmetry_index_pct(a: float, b: float) -> float | None:
    """|a - b| / (0.5 * (a + b)) * 100, the standard gait symmetry index.

    Not defined for values whose mean is ~0 (a tilt hovering around zero), where it
    explodes; None is returned instead of a meaningless number.
    """
    if a is None or b is None:
        return None
    m = 0.5 * (abs(a) + abs(b))
    if not np.isfinite(m) or m < 1e-6:
        return None
    return float(abs(a - b) / m * 100.0)
