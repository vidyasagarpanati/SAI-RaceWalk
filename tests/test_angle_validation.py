"""Validates every joint angle S3 computes (knee, hip, ankle, elbow, both legs) against the
exact noiseless geometry the synthetic walker was built from (tests/synthetic.py::true_angles).

This is independent of the stride/event tests: it checks the angle FORMULA and landmark
choices are correct, not gait detection. A wrong vertex or a swapped proximal/distal point
would show up here as a large, systematic error, not just noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import helpers
import synthetic
from racewalk.steps import s03_kinematics

ANGLE_COLS = ["knee_left_deg", "knee_right_deg", "hip_left_deg", "hip_right_deg",
              "ankle_left_deg", "ankle_right_deg", "elbow_left_deg", "elbow_right_deg"]


@pytest.fixture(scope="module")
def measured_vs_true(tmp_path_factory):
    base = tmp_path_factory.mktemp("angle_check")
    df, truth = synthetic.build(duration_s=20)
    ctx = helpers.make_run(base / "run", df, truth["fps"])
    res = s03_kinematics.run(ctx)
    assert res.passed, [f"{c.name}: {c.detail}" for c in res.failures]
    kin = pd.read_parquet(ctx.artefact("03_kinematics.parquet"))
    # Trim the smoothing edge (half the Savitzky-Golay window at each end) and stride
    # transition frames, where a phase-lagged smoothed value is expected to differ from the
    # instantaneous truth by more than noise.
    edge = 5
    return kin.iloc[edge:-edge].reset_index(drop=True), {k: v[edge:-edge] for k, v in truth["true_angles"].items()}


def test_every_angle_column_is_computed(measured_vs_true):
    kin, _ = measured_vs_true
    for col in ANGLE_COLS:
        assert col in kin.columns, f"{col} is missing from 03_kinematics.parquet"


@pytest.mark.parametrize("col", ANGLE_COLS)
def test_angle_matches_ground_truth(measured_vs_true, col):
    kin, true = measured_vs_true
    measured = kin[col].to_numpy(float)
    truth = true[col]
    ok = np.isfinite(measured) & np.isfinite(truth)
    assert ok.mean() > 0.95, f"{col}: too many withheld/NaN frames to validate ({ok.mean():.0%} usable)"
    err = np.abs(measured[ok] - truth[ok])
    # Savitzky-Golay smoothing (window=9) lags fast-changing angles (knee/ankle in swing);
    # a static angle (elbow, hip here) should match almost exactly.
    # Knee sweeps ~45 deg during swing-to-stance, the fastest-changing angle measured, so
    # smoothing lag costs it a bit more than the near-static hip/elbow/ankle angles.
    limit = 3.5 if col.startswith("knee") else 2.0
    assert np.median(err) < limit, f"{col}: median error {np.median(err):.2f} deg (systematic bias, likely wrong landmarks)"
    assert np.percentile(err, 95) < 8.0, f"{col}: 95th-percentile error {np.percentile(err, 95):.2f} deg"


def test_elbow_angle_is_stable_and_not_swapped_with_shoulder(measured_vs_true):
    """A wrong vertex (e.g. elbow angle computed at the shoulder) would show as a large,
    obviously-wrong mean rather than noise around the true ~89 deg."""
    kin, true = measured_vs_true
    for lg in ("left", "right"):
        col = f"elbow_{lg}_deg"
        measured = kin[col].to_numpy(float)
        measured = measured[np.isfinite(measured)]
        assert 70 < measured.mean() < 110, f"{col}: mean {measured.mean():.1f} deg is not near the true ~89 deg"
        assert measured.std() < 5, f"{col}: SD {measured.std():.1f} deg is too high for a near-constant swing angle"
