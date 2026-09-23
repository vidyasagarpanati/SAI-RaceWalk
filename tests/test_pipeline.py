"""S3 to S10 end to end on a synthetic walker with known ground truth, and a fake model."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

import helpers
import synthetic
from racewalk.llm import FakeLLM
from racewalk.steps import (s03_kinematics, s04_gait, s05_stats, s06_annotate, s07_video,
                            s08_narrate, s09_verify, s10_render)

BENT = {"L": {5: 25, 6: 25}, "R": {20: 25}}


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    base = tmp_path_factory.mktemp("rw")
    df, truth = synthetic.build(duration_s=40, bent=BENT, flight_at_s=25.0, drift=True)
    ctx = helpers.make_run(base / "run", df, truth["fps"], outputs_dir=base / "out")
    ctx.llm = FakeLLM()
    res = {}
    for name, step in (("s3", s03_kinematics), ("s4", s04_gait), ("s5", s05_stats), ("s6", s06_annotate),
                       ("s7", s07_video), ("s8", s08_narrate), ("s9", s09_verify), ("s10", s10_render)):
        res[name] = step.run(ctx)
    metrics = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    return ctx, base, res, metrics, truth


def test_every_step_passes(full):
    _, _, res, *_ = full
    for name, r in res.items():
        assert r.passed, (name, [f"{c.name}: {c.detail}" for c in r.failures])


def test_events_match_ground_truth(full):
    ctx, _, res, m, truth = full
    g = json.loads((ctx.run_dir / "04_gait.json").read_text())
    assert 24 <= g["n_strides_by_leg"]["L"] <= 53 and g["heel_strike_alternation_share"] >= 0.95
    for leg in "LR":
        hs = np.array([c[0] / 60 for c in g["legs"][leg]["contacts"]])
        true = np.array(truth[f"hs_{leg}_s"])
        err = [float(np.min(np.abs(true - x))) for x in hs[1:-1]]
        assert max(err) < 0.20       # detector fires early on the synthetic's shallow foot approach; drift widens it
        to = np.array([c[1] / 60 for c in g["legs"][leg]["contacts"]])
        true_to = np.array(truth[f"to_{leg}_s"])
        assert max(float(np.min(np.abs(true_to - x))) for x in to[1:-1]) < 0.06


def test_straight_leg_screen_flags_exactly_the_bent_strides(full):
    _, _, _, m, _ = full
    ids = sorted(sid for e in m["events"] if e["type"] == "STRAIGHT_LEG" for sid in e["stride_ids"])
    assert ids == ["L06", "L07", "R21"]
    assert m["gait"]["straight_leg_flagged"] == 3


def test_flight_episode_found_at_injected_time(full):
    _, _, _, m, _ = full
    fl = [e for e in m["events"] if e["type"] == "CONTACT"]
    assert len(fl) == 1 and abs(fl[0]["start_t_s"] - 25.0) < 0.1


def test_rear_view_caps_knee_confidence_at_low(full):
    _, _, _, m, _ = full
    assert m["stride_stats"]["both"]["loading_knee_min_deg"]["confidence"] == "LOW"
    assert "gait.heel_first_contact_pct" not in m["evidence_index"]      # side-only measure withheld
    assert "NOT RELIABLY ASSESSABLE" in m["technique"]["foot_strike"]["status"]


def test_speed_never_estimated_without_belt_speed(full):
    _, _, _, m, _ = full
    ev = m["evidence_index"]
    assert "gait.pace_min_per_km" not in ev and "gait.step_length_m" not in ev


def test_fatigue_drift_detected(full):
    _, _, _, m, _ = full
    assert m["drift"]["available"]
    assert any(e["type"] == "FATIGUE" for e in m["events"])


def test_metrics_are_reproducible(tmp_path):
    df, truth = synthetic.build(duration_s=20, bent=BENT)
    out = []
    for i in range(2):
        ctx = helpers.make_run(tmp_path / f"r{i}", df, truth["fps"])
        for st in (s03_kinematics, s04_gait, s05_stats):
            assert st.run(ctx).passed
        out.append((ctx.run_dir / "05_metrics.json").read_text().replace(ctx.run_id, "X"))
    assert out[0] == out[1]


def test_report_is_self_contained_versioned_and_grounded(full):
    ctx, base, *_ = full
    reports = sorted((base / "out").glob("RaceWalk_Report_*_v*.html"))
    assert reports and reports[0].name.endswith("_v01.html")
    html = reports[0].read_text()
    assert "{{" not in html and not re.search(r'(src|href)\s*=\s*["\']https?:', html)
    n_frames = len(json.loads((ctx.run_dir / "06_manifest.json").read_text())["frames"])
    assert html.count("data:image/jpeg;base64,") == n_frames >= 15
    assert "Individualized assessment required" in html and "Generic guidance" in html
    ids = re.findall(r'<section id="sec-(\w+)"', html)
    assert ids == ["info"] + [f"s0{i}" for i in range(1, 10)] + ["prov"]


def test_annotated_video_and_event_clips_written(full):
    ctx, base, *_ = full
    assert (ctx.run_dir / "07_video" / "annotated_full.mp4").is_file()
    assert list((base / "out").glob("*_Annotated_*_v01.mp4"))
    assert len(list((ctx.run_dir / "07_video").glob("ev*.mp4"))) >= 3


def test_no_measured_number_can_be_typed_by_the_model(tmp_path):
    df, truth = synthetic.build(duration_s=20, bent=BENT)
    ctx = helpers.make_run(tmp_path / "run", df, truth["fps"], outputs_dir=tmp_path / "out")
    for st in (s03_kinematics, s04_gait, s05_stats, s06_annotate):
        st.run(ctx)
    ctx.llm = FakeLLM(fabricate_in="s02_technique")
    r8 = s08_narrate.run(ctx)
    log = json.loads((ctx.narrative_dir / "generation_log.json").read_text())
    assert any("typed outside" in p for e in log if e["section"] == "s02_technique" for p in e["problems"])
    assert r8.passed          # the retry, with the violation listed, succeeded


def test_truncated_reply_is_retried(tmp_path):
    df, truth = synthetic.build(duration_s=20)
    ctx = helpers.make_run(tmp_path / "run", df, truth["fps"], outputs_dir=tmp_path / "out")
    for st in (s03_kinematics, s04_gait, s05_stats, s06_annotate):
        st.run(ctx)
    ctx.llm = FakeLLM(truncate_in="s03_injury")
    assert s08_narrate.run(ctx).passed


def test_side_view_measures_step_length_and_foot_pitch(tmp_path):
    df, truth = synthetic.build(duration_s=20, view="side")
    ctx = helpers.make_run(tmp_path / "run", df, truth["fps"], session={"camera_view": "side_left"})
    for st in (s03_kinematics, s04_gait, s05_stats):
        assert st.run(ctx).passed
    m = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    assert "both.stride.step_length_leg.mean" in m["evidence_index"]
    assert m["stride_stats"]["both"]["loading_knee_min_deg"]["confidence"] in ("MEDIUM", "HIGH")
    assert m["risk"]["ankle_stability"]["level"] == "NOT ASSESSED"      # frontal-only signals
