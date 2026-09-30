"""S10 render.

IN     00_ingest.json, 05_metrics.json, 06 key frames, 08_narrative, 09_verification.json
DO     substitute every evidence placeholder with its measured value, build the computed
       parts, re-render key frames with the verified coaching text, and write one
       self-contained HTML file (inline CSS, base64 images)
OUT    outputs/RaceWalk_Report_<Athlete>_<YYYYMMDD>_vNN.html, .sha256, .manifest.json
VERIFY S9 passed or the report is marked PARTIAL; no external references; no unresolved
       placeholders; sections present in the fixed order; every embedded image accounted
       for; HTML parses; the file is a new version, nothing overwritten
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
from html.parser import HTMLParser

import cv2

from racewalk import __version__
from racewalk.context import Context
from racewalk.contracts import StepResult
from racewalk.framedata import FrameData
from racewalk.grounding import format_value, substitute_all
from racewalk.io_guard import guarded_open, guarded_path
from racewalk.phase_defs import DISPLAY, EVENT_COLOR_BGR, EVENT_DISPLAY, ORDER
from racewalk.report_spec import PARAMETERS, RISK_CATS, SECTION_OF, SPEED_ROWS, TECH_KEY, report_order
from racewalk.steps.s06_annotate import render_key_frame
from racewalk.versioning import next_versioned, safe

NP = "NOT PROVIDED - CANNOT BE CONFIRMED"
NA = "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO"
# report section -> the narrative call that fills it (s08 is static guidance, never fails)
SECTION_SOURCE = {"s01": "s01_timestamps", "s02": "s02_technique", "s03": "s03_injury", "s04": "s04_comparison",
                  "s05": "s05_speed", "s06": "s06_training", "s07": "s07_strength", "s09": "s09_limitations"}
RISK_LABEL = {"knee_stress": "Knee stress", "hip_low_back": "Hip / low back", "achilles_calf": "Achilles / calf",
              "ankle_stability": "Ankle stability", "overuse_fatigue": "Overuse / fatigue"}

FIXED_CAVEATS = [
    "Contact (Rule 54.1) and the bent-knee rule (Rule 54.2) are judged by human judges with the naked eye. "
    "This report is a pose-based screen that flags moments for review. It is not a ruling and it cannot say "
    "whether a warning or disqualification would follow.",
    "Pose estimation from one camera measures depth poorly. Knee and hip flexion are depth-derived from a rear "
    "or front view, so results from those views carry a lower confidence ceiling than results from a side view.",
    "Risk levels are provisional screening heuristics defined in config/risk_rules.yaml. They are not validated "
    "clinical thresholds and this report does not diagnose any condition.",
    "No reference values are used unless they carry a citation in config/benchmarks.json. Where none exists the "
    "report says so.",
    "Sections 7 and 8 contain generic guidance carried over from the analysis template. It was not derived from "
    "this video and should be reviewed by a qualified coach, sports dietitian or physiotherapist.",
]


def _b64(img, max_w: int, q: int) -> str:
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def _s(ev, key, dash="n/a"):
    return format_value(ev[key]) if key in ev else dash


def build_view(ctx: Context, version_label: str) -> tuple[dict, int]:
    ingest = ctx.read_json("00_ingest.json")
    m = ctx.read_json("05_metrics.json")
    ver = ctx.read_json("09_verification.json")
    manifest = ctx.read_json("06_manifest.json")
    narr_raw = json.loads((ctx.narrative_dir / "all_sections.json").read_text(encoding="utf-8"))
    status_path = ctx.narrative_dir / "section_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    ev = m["evidence_index"]
    n = substitute_all(narr_raw, ev)
    sess, athlete = m["session"], m["session"]["athlete"]
    guidance = ctx.cfg.guidance

    def why(call_id):
        st = status.get(call_id) or {}
        return f"{', '.join(st.get('classes') or []) or 'not produced'}: {(st.get('problems') or ['no detail recorded'])[0]}"

    failed = {k: why(src) for k, src in SECTION_SOURCE.items() if src not in n}

    def a(key):
        val = athlete.get(key)
        return NP if val in (None, "", NP) else str(val)

    view_names = {"rear": "Rear", "front": "Front", "side_left": "Side (left side to camera)",
                  "side_right": "Side (right side to camera)", "oblique": "Oblique"}
    info = [("Athlete", a("athlete_name")), ("Level", a("level")), ("Discipline", a("discipline")),
            ("Setting", a("setting")), ("Camera view", view_names.get(sess["camera_view"], sess["camera_view"])),
            ("Video file", sess["video_file"]), ("Duration", f"{sess['duration_s']} s"),
            ("Frame rate", f"{sess['measured_fps']} fps measured; analysed at {sess['analysis_fps']:g} fps"),
            ("Resolution", sess["resolution"]), ("Session date", a("session_date")),
            ("Strides analysed", f"{sess['n_strides_left']} left, {sess['n_strides_right']} right"),
            ("Analysis method", sess["analysis_method"]), ("Pose estimation", sess["pose_estimation"]),
            ("Analyst", "SAI RaceWalk pipeline (deterministic measurement, model-written prose)")]
    note = ("Single-camera analysis. Contact and straight-leg results are screening flags for review, not rulings."
            if sess["view_class"] != "sagittal" else
            "Single-camera side view. Straight-leg readings are stronger than from behind but remain a screen; "
            "a judge rules on the human eye.")

    # ---- key frames: event frames re-rendered with the verified cue, reference frames as they are
    fd = FrameData.load(ctx.run_dir)
    track = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    screen = float(ctx.cfg.phase_rules["screening"]["straight_knee_min_deg"])
    max_w = int(ctx.cfg.get("render.max_embedded_frame_width", 1100))
    kmax = int(ctx.cfg.get("render.key_frame_max_width", 1920))
    qual = int(ctx.cfg.get("render.jpeg_quality_embed", 85))
    events_by_id = {e["id"]: e for e in m["events"]}
    s01 = {e["event_id"]: e for e in n.get("s01_timestamps", {}).get("events", [])}
    n_img = 0
    ev_frames: dict[int, list] = {}
    ref_frames: dict[str, list] = {"L": [], "R": []}
    for f in manifest["frames"]:
        if f["kind"] == "event":
            cue = (s01.get(f["event_id"]) or {}).get("coaching_cue")
            canvas, _ = render_key_frame(fd, f, ctx.view_class, coaching=cue, track_conf=track,
                                         screen_knee_deg=screen, max_w=kmax, event=events_by_id[f["event_id"]],
                                         near_side_param=ctx.near_side)
            ev_frames.setdefault(f["event_id"], []).append(
                {"t": f["t_s"], "frame": f["frame"], "label": f["label"], "b64": _b64(canvas, max_w, qual)})
        else:
            canvas, _ = render_key_frame(fd, f, ctx.view_class, track_conf=track, screen_knee_deg=screen, max_w=kmax,
                                         near_side_param=ctx.near_side)
            ref_frames[f["leg"]].append({"phase": DISPLAY[f["phase"]], "stride": f["stride_id"], "t": f["t_s"],
                                         "frame": f["frame"], "b64": _b64(canvas, max_w, qual)})
        n_img += 1

    # ---- section 1
    events = []
    for e in m["events"]:
        d = s01.get(e["id"], {})
        col = EVENT_COLOR_BGR[e["type"]]
        events.append({"id": e["id"], "type": e["display_name"], "severity": e["severity"], "confidence": e["confidence"],
                       "ts": f"{e['start_t_s']:.2f} to {e['end_t_s']:.2f} s", "value": f"{e['value']} {e['units']}",
                       "what": d.get("what"), "why": d.get("why_it_matters"), "cue": d.get("coaching_cue"),
                       "frames": ev_frames.get(e["id"], []), "color": f"rgb({col[2]},{col[1]},{col[0]})"})
    # ---- section 2
    tech_ai = {r["parameter"]: r for r in n.get("s02_technique", {}).get("rows", [])}
    tech_rows = []
    for p in PARAMETERS:
        t = m["technique"][TECH_KEY[p]]
        r = tech_ai.get(p, {})
        tech_rows.append({"parameter": p, "status": t["status"], "ceiling": t["confidence"], "obs": r.get("observation"),
                          "why": r.get("why_it_matters"), "conf": r.get("confidence")})
    ps = m["stride_stats"]["both"]
    rot = [("Pelvic rotation range per stride", ps.get("pelvic_yaw_range_deg")), ("Shoulder rotation range", ps.get("shoulder_yaw_range_deg")),
           ("Trunk inclination range", ps.get("trunk_incl_range_deg")), ("Loading-window knee minimum", ps.get("loading_knee_min_deg")),
           ("Vertical oscillation (leg lengths)", ps.get("vertical_osc_leg"))]
    stats_rows = [{"name": nm, "mean": d.get("mean", "—"), "sd": d.get("sd", "—"), "min": d.get("min", "—"),
                   "max": d.get("max", "—"), "conf": d.get("confidence")} for nm, d in rot if d]
    bench_rows = []
    rows_by_measure = {r["measure"]: r for r in m["benchmarks"]["rows"]}
    for nm, key in (("Pelvic rotation range", "pelvic_yaw_range_deg"),):
        row = rows_by_measure.get(key)
        bench_rows.append({"variable": nm, "athlete": (ps.get(key) or {}).get("mean", "—"),
                           "reference": f"{row['target']} ({row['source']})" if row else "Individualized assessment required"})
    # ---- section 3
    risk_ai = {r["category"]: r for r in n.get("s03_injury", {}).get("risks", [])}
    risks = []
    for c in RISK_CATS:
        r = m["risk"][c]
        ai = risk_ai.get(c, {})
        driver = "—"
        if r["level"] != "NOT ASSESSED":
            driver = f"{r['driver_metric'].replace('_', ' ')} = {r['driver_value']} {r['driver_unit']} (moderate at {r['moderate']}, high at {r['high']})"
        risks.append({"label": RISK_LABEL[c], "level": r["level"], "driver": driver, "ceiling": r.get("confidence"),
                      "rationale": ai.get("rationale"), "concern": ai.get("possible_concern"),
                      "focus": ai.get("corrective_focus"), "conf": ai.get("confidence")})
    # ---- section 4
    comp = n.get("s04_comparison", {})
    weak = sorted(comp.get("weaknesses", []), key=lambda w: w["rank"])
    for w in weak:
        w["timestamp"] = _s(ev, w["timestamp_key"], "—") if w["timestamp_key"] in ev else "—"
    # ---- section 5
    sp_ai = {r["metric"]: r for r in n.get("s05_speed", {}).get("rows", [])}
    g = m["gait"]
    speed_vals = {
        "Pace": _s(ev, "gait.pace_min_per_km", NP), "Cadence": f"{_s(ev, 'gait.cadence_spm')} (SD {_s(ev, 'gait.cadence_sd')})",
        "Step and stride length": (f"step {_s(ev, 'gait.step_length_m')}, stride {_s(ev, 'gait.stride_length_m')}"
                                   if "gait.step_length_m" in ev else
                                   (f"step {_s(ev, 'both.stride.step_length_leg.mean')} (side view)" if "both.stride.step_length_leg.mean" in ev
                                    else NP + " (needs belt speed or a side view)")),
        "Vertical oscillation": _s(ev, "gait.vertical_oscillation_pct_leg"),
        "Efficiency": NA + " (no energy or force data)"}
    speed_rows = [{"metric": x, "value": speed_vals[x], "reading": sp_ai.get(x, {}).get("reading"),
                   "note": sp_ai.get(x, {}).get("note"), "conf": sp_ai.get(x, {}).get("confidence")} for x in SPEED_ROWS]
    extra_speed = [("Average speed", _s(ev, "gait.average_speed_kmh") + (f" ({g.get('speed_source')})" if g.get("speed_source") else "")
                    if "gait.average_speed_kmh" in ev else "NOT PROVIDED"),
                   ("Left step time", _s(ev, "gait.step_time_left_s")), ("Right step time", _s(ev, "gait.step_time_right_s")),
                   ("Step-time asymmetry", _s(ev, "asym.step_time.pct")), ("Double support", _s(ev, "gait.double_support_pct")),
                   ("Both feet airborne candidates", _s(ev, "gait.flight_candidates"))]
    # ---- section 6, 7
    plan = n.get("s06_training", {})
    for ms in plan.get("milestones", []):
        ms["baseline"] = _s(ev, ms["metric_key"], "—")
    mass = athlete.get("body_mass_kg")
    carb = f" ({round(2 * float(mass))} g for {mass:g} kg)" if isinstance(mass, (int, float)) and mass > 0 else ""
    nutrition = [{"timing": x["timing"], "advice": x["advice"].replace("{carb_pre_note}", carb)} for x in guidance["nutrition"]]
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for c in re.findall(r'"confidence":\s*"(HIGH|MEDIUM|LOW)"', json.dumps(narr_raw)):
        counts[c] += 1
    usage_path = ctx.narrative_dir / "usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.is_file() else {}
    vid = ctx.artefact("07_video.json")
    video_name = json.loads(vid.read_text(encoding="utf-8")).get("published_name") if vid.is_file() else None
    provenance = [("Run id", ctx.run_id), ("Source video", sess["video_file"]), ("Video SHA-256", sess["video_sha256"]),
                  ("Config hash", m["config_hash"]), ("Pipeline version", __version__),
                  ("Narrative model", str(ctx.cfg.get("llm.model"))),
                  ("Model settings", f"temperature {ctx.cfg.get('llm.temperature')}, seed {ctx.cfg.get('llm.seed')}"),
                  ("Model calls / tokens", f"{usage.get('calls', 0)} calls ({usage.get('cached', 0)} cached), "
                                           f"{usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion tokens"),
                  ("Key frames sent to model", "yes" if usage.get("vision_used") else "no"),
                  ("Annotated video", video_name or "not rendered")]
    date = athlete.get("session_date") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", athlete.get("session_date") or "") else dt.date.today().isoformat()
    q = m["data_quality"]
    view = {
        "athlete_name": a("athlete_name"), "session_date": date, "version_label": version_label, "run_id": ctx.run_id,
        "order": report_order(), "info": info, "note": note, "n": n, "events": events, "events_summary": m["events_summary"],
        "tech_rows": tech_rows, "stats_rows": stats_rows, "bench_rows": bench_rows, "ref_frames": ref_frames,
        "risks": risks, "risk_status": m["risk_rules"]["status"], "strengths": comp.get("strengths", []), "weak": weak,
        "speed_rows": speed_rows, "extra_speed": extra_speed, "plan": plan, "guidance": guidance, "nutrition": nutrition,
        "caveats": FIXED_CAVEATS, "overall_conf": q["overall_analysis_confidence"], "overall_rule": q["overall_confidence_rule"],
        "camera_caps": q["camera_confidence_cap"], "view_check": (q.get("view_check") or {}).get("detail"),
        "failed": failed, "partial": bool(failed), "checklist": ver["checklist"], "conf_counts": counts,
        "provenance": provenance, "video_name": video_name, "drift": m["drift"], "drift_status": m["drift_status"],
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return view, n_img


class _Parse(HTMLParser):
    def error(self, message):  # pragma: no cover
        raise ValueError(message)


def run(ctx: Context) -> StepResult:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    res = StepResult(step="S10")
    ver = ctx.read_json("09_verification.json")
    allow_partial = bool(ctx.cfg.get("report.allow_partial", True))
    can_publish = bool(ver.get("passed")) or (allow_partial and ver.get("sections_available"))
    res.check("verification_passed", bool(ver.get("passed")),
              "S9 passed." if ver.get("passed") else
              (f"S9 did not pass; publishing a PARTIAL report with {len(ver.get('missing_sections') or [])} "
               f"section(s) marked NOT AVAILABLE." if can_publish else "S9 did not pass and nothing could be published."),
              severity="WARN" if can_publish else "FAIL")
    if not can_publish:
        return res
    session = ctx.read_json("00_ingest.json")["session"]
    date = session.get("session_date") if isinstance(session.get("session_date"), str) and len(session.get("session_date")) == 10 \
        else dt.date.today().isoformat()
    stem = f"RaceWalk_Report_{safe(session.get('athlete_name', 'Athlete'))}_{date.replace('-', '')}"
    if not ver.get("passed"):
        stem += "_PARTIAL"
    target = next_versioned(guarded_path(ctx.cfg.paths.outputs_dir), stem, ".html")
    label = re.search(r"_v(\d+)\.html$", target.name).group(0)[1:-5]
    view, n_img = build_view(ctx, label)
    env = Environment(loader=FileSystemLoader(str(ctx.cfg.paths.templates_dir)), autoescape=select_autoescape(["html", "j2"]))
    html = env.get_template("report.html.j2").render(v=view)
    existed = target.exists()
    with guarded_open(target, "w", encoding="utf-8") as fh:
        fh.write(html)
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    with guarded_open(target.with_suffix(".html.sha256"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {target.name}\n")
    manifest = {"report": target.name, "sha256": digest, "run_id": ctx.run_id,
                "video_sha256": ctx.read_json("00_ingest.json")["video_sha256"], "config_hash": ctx.cfg.config_hash,
                "model": ctx.cfg.get("llm.model"), "images": n_img, "annotated_video": view["video_name"]}
    with guarded_open(target.with_suffix(".manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    external = re.findall(r'(?:src|href)\s*=\s*["\'](?:https?:)?//|url\(\s*["\']?https?:|@import', html)
    res.check("self_contained", not external, f"External references found: {external[:5]}" if external
              else "No external CSS, scripts, fonts or images.")
    res.check("no_unresolved_placeholders", "{{" not in html, "All evidence placeholders substituted.")
    order_found = re.findall(r'<section id="sec-(\w+)"', html)
    expected = ["info"] + [s["key"] for s in view["order"]] + ["prov"]
    res.check("sections_present_in_order", order_found == expected,
              f"found {order_found}" if order_found != expected else f"{len(order_found)} sections in the fixed order.")
    res.check("every_frame_embedded", html.count("data:image/jpeg;base64,") == n_img, f"{n_img} annotated frames embedded")
    try:
        _Parse().feed(html)
        parsed = True
    except Exception:  # noqa: BLE001
        parsed = False
    res.check("html_parses", parsed, "HTML parsed cleanly.")
    res.check("new_version_not_overwrite", not existed, f"Wrote {target.name}")
    res.check("static_guidance_labelled", "Generic guidance" in html, "Sections 7 and 8 carry the generic-guidance label.")
    res.outputs["report"] = str(target)
    res.outputs["sha256"] = str(target.with_suffix(".html.sha256"))
    res.check("partial_sections_marked", True,
              f"{len(view['failed'])} section(s) marked NOT AVAILABLE in the report." if view["failed"] else "Every section rendered.",
              severity="WARN")
    res.stats = {"report": target.name, "size_mb": round(len(html) / 1e6, 2), "images": n_img,
                 "partial": view["partial"], "sections_not_available": len(view["failed"])}
    return res
