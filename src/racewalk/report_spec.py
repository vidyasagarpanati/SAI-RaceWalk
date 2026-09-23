"""Report structure, section schemas, context builders and section self-checks.

The report's shape is fixed here, in code, so it cannot drift between runs: section order,
names, which parts are computed in Python and which are written by the model, what each
model call may see, and the rules each output must satisfy.

What Python decides, and the model never does:
  * every number (05_metrics.json)
  * the event list and each event's severity and confidence (S5 screening rules)
  * the status of each technique row and the level of each injury-risk category
  * everything in Sections 7 and 8 that is generic guidance (config/guidance.yaml)

What the model writes: the prose that explains those results and the coaching plan built
on them, citing evidence keys, never typing a measured number.

Token economy: the model never receives the video, the landmark table or the whole metrics
file. Each call gets the shared rules, one section's instructions, its schema, and an
EVIDENCE list of only the keys that section needs. Later sections receive the earlier
sections' JSON instead of raw metrics, which also keeps the report internally consistent.
"""
from __future__ import annotations

import json
import re
from typing import Any

from racewalk.grounding import format_value, keys_in, walk_strings

NA = "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO"
LEVELS = ["OBSERVED", "MEASURED", "INTERPRETED", "INFERRED"]
CONF = ["HIGH", "MEDIUM", "LOW"]
SEV = ["CRITICAL", "MODERATE", "MINOR"]
NOT_ESTABLISHED = "NOT ESTABLISHED"

PARAMETERS = ["Contact", "Straight leg", "Pelvic rotation", "Torso posture",
              "Stride length and frequency", "Foot strike"]
TECH_KEY = dict(zip(PARAMETERS, ["contact", "straight_leg", "pelvic_rotation", "torso_posture", "stride", "foot_strike"]))
RISK_CATS = ["knee_stress", "hip_low_back", "achilles_calf", "ankle_stability", "overuse_fatigue"]
SPEED_ROWS = ["Pace", "Cadence", "Step and stride length", "Vertical oscillation", "Efficiency"]
PLAN_BLOCKS = ["Weeks 1-4", "Weeks 5-8", "Weeks 9-12"]
DAYS = ["Session A", "Session B", "Session C"]

# ------------------------------------------------------------------ report order
MASTER = [("s01", "TIMESTAMPS OF INTEREST"), ("s02", "TECHNIQUE BREAKDOWN"),
          ("s03", "INJURY RISK ASSESSMENT"), ("s04", "COMPARISON TO IDEAL FORM"),
          ("s05", "SPEED AND EFFICIENCY METRICS"), ("s06", "COACHING SUGGESTIONS AND TRAINING PLAN"),
          ("s07", "STRENGTH AND CONDITIONING SUGGESTIONS"), ("s08", "SPORTS SCIENCE RELATED SUGGESTIONS"),
          ("s09", "LIMITATIONS AND CAVEATS")]


def report_order() -> list[dict]:
    return [{"key": k, "number": str(i), "title": t} for i, (k, t) in enumerate(MASTER, 1)]


# ------------------------------------------------------------------ schema helpers
STR = {"type": "string", "maxLength": 900}
INT = {"type": "integer"}
KEY_STR = {"type": "string", "pattern": "^[A-Za-z0-9_.\\-]+$", "maxLength": 120}


def enum(vals):
    return {"type": "string", "enum": list(vals)}


def arr(item, mn=0, mx=None):
    a = {"type": "array", "items": item, "minItems": mn}
    if mx is not None:
        a["maxItems"] = mx
    return a


def obj(**props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


# Only fields that hold ids, evidence keys or closed enums are exempt from grounding.
# Every free-text field is grounded, however short.
SKIP_FIELDS = {"evidence_level", "confidence", "severity", "category", "parameter", "metric", "id", "ref",
               "issue_ref", "priority", "evidence_keys", "addresses", "event_id", "timestamp_key", "phase",
               "day", "when", "rank", "metric_key"}
# Training doses live here. Numbers are allowed (sets, minutes, distances), but any number
# that looks like a MEASUREMENT (degrees, percent, cadence, speed) is rejected by local_checks.
PRESCRIPTIVE = {"sets", "frequency", "drill", "dose", "content", "success_metric", "target", "tracked_by",
                "cue", "exercise"}
MEASUREMENT_LIKE = re.compile(r"\d+(?:\.\d+)?\s*(?:°|deg\b|degrees?|%|percent|steps?/min|spm|km/h|min/km|m/s)", re.I)


def valid_ids(outputs: dict) -> dict[str, list[str]]:
    c = outputs.get("s04_comparison", {})
    return {"strengths": [i["id"] for i in c.get("strengths", [])],
            "weaknesses": [i["id"] for i in c.get("weaknesses", [])]}


def _ref_enum(values: list[str]) -> dict:
    return enum(list(dict.fromkeys(values + [NOT_ESTABLISHED])))


def schema_for(sid: str, ctx: dict) -> dict:
    ids = ctx.get("ids") or {"strengths": [], "weaknesses": []}
    call_keys = ctx.get("evidence_keys")
    KEYS = arr(enum(call_keys), 0, 6) if call_keys else arr(KEY_STR, 0, 6)
    event_ids = ctx.get("event_ids") or []
    W_ENUM = _ref_enum(ids["weaknesses"])
    ID_ENUM = {"s04_comparison_s": enum([f"S{i}" for i in range(1, 9)]),
               "s04_comparison_w": enum([f"W{i}" for i in range(1, 9)])}
    if sid == "s09_limitations":
        return obj(camera=STR, lighting=STR, occlusion=STR, motion_blur=STR, clothing=STR,
                   additional=arr(STR, 1, 6))
    if sid == "s01_timestamps":
        n = len(event_ids)
        return obj(summary=STR,
                   events=arr(obj(event_id={"type": "integer", "enum": event_ids or [0]}, what=STR,
                                  why_it_matters=STR, coaching_cue=STR, confidence=enum(CONF),
                                  evidence_keys=KEYS), n, n))
    if sid == "s02_technique":
        return obj(rows=arr(obj(parameter=enum(PARAMETERS), observation=STR, why_it_matters=STR,
                                confidence=enum(CONF), evidence_keys=KEYS), 6, 6))
    if sid == "s03_injury":
        return obj(risks=arr(obj(category=enum(RISK_CATS), rationale=STR, possible_concern=STR,
                                 corrective_focus=STR, confidence=enum(CONF), evidence_keys=KEYS), 5, 5))
    if sid == "s04_comparison":
        tsk = enum(list(dict.fromkeys(ctx.get("timestamp_keys", []) + [NOT_ESTABLISHED])))
        return obj(strengths=arr(obj(id=ID_ENUM["s04_comparison_s"], strength=STR, evidence=STR,
                                     confidence=enum(CONF), evidence_keys=KEYS), 2, 6),
                   weaknesses=arr(obj(id=ID_ENUM["s04_comparison_w"], rank=INT, weakness=STR, evidence=STR,
                                      likely_cause=STR, correction=STR, timestamp_key=tsk, priority=enum(SEV),
                                      confidence=enum(CONF), evidence_keys=KEYS), 2, 6))
    if sid == "s05_speed":
        return obj(rows=arr(obj(metric=enum(SPEED_ROWS), reading=STR, note=STR, confidence=enum(CONF),
                                evidence_keys=KEYS), 5, 5), summary=STR)
    if sid == "s06_training":
        return obj(immediate=arr(obj(rank=INT, issue_ref=W_ENUM, cue=STR, drill=STR, sets=STR, frequency=STR,
                                     success_metric=STR), 3, 5),
                   phases=arr(obj(phase=enum(PLAN_BLOCKS), focus=STR,
                                  sessions=arr(obj(day=enum(DAYS), content=STR), 3, 3)), 3, 3),
                   milestones=arr(obj(when=enum(PLAN_BLOCKS), metric_key=KEY_STR, target=STR, tracked_by=STR), 3, 3))
    if sid == "s07_strength":
        return obj(exercises=arr(obj(area=STR, exercise=STR, dose=STR, purpose=STR,
                                     addresses=arr(W_ENUM, 1, 3)), 4, 8), notes=STR)
    raise KeyError(sid)


# ------------------------------------------------------------------ output template
def _example(node: dict, depth: int = 0):
    t = node.get("type")
    if "enum" in node:
        vals = node["enum"]
        return vals[0] if len(vals) == 1 else " | ".join(map(str, vals[:12])) + (" | ..." if len(vals) > 12 else "")
    if t == "object":
        return {k: _example(v, depth + 1) for k, v in node["properties"].items()}
    if t == "array":
        return [_example(node["items"], depth + 1)]
    if t == "integer" or t == ["integer", "null"]:
        return "<integer>"
    if node.get("pattern"):
        return "<evidence key, exactly as listed, WITHOUT braces>"
    return "<text>"


def _counts(node: dict, path: str = "") -> list[str]:
    out = []
    if node.get("type") == "object":
        for k, v in node["properties"].items():
            out += _counts(v, f"{path}.{k}" if path else k)
    elif node.get("type") == "array":
        mn, mx = node.get("minItems", 0), node.get("maxItems")
        if mn == mx:
            out.append(f"{path}: exactly {mn} item(s)")
        elif mn or mx is not None:
            out.append(f"{path}: {mn} to {mx if mx is not None else 'any'} items")
        out += _counts(node["items"], f"{path}[]")
    return out


def output_format(schema: dict) -> str:
    """Human-readable template of the required JSON, derived from the schema so the two can
    never disagree."""
    template = json.dumps(_example(schema), indent=1, ensure_ascii=False)
    lines = [
        "OUTPUT FORMAT",
        "Return ONE JSON object with exactly these fields and nothing else. No markdown,",
        "no comments, no text before or after it. Values separated by | are the only",
        "allowed choices; pick one.",
        template,
        "Item counts:",
        *[f"- {c}" for c in _counts(schema)],
        "Be concise: one or two sentences per text field. In text fields cite values as",
        "{{KEY}}. In evidence_keys arrays list bare keys WITHOUT braces, at most 6.",
        "Finish the JSON completely; a cut-off answer is rejected.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ call plan
# (section id, depends on earlier outputs, which key frames to attach)
CALLS: list[tuple[str, list[str], list[str]]] = [
    ("s09_limitations", [], ["reference"]),
    ("s01_timestamps", [], ["event"]),
    ("s02_technique", ["s01_timestamps"], ["reference"]),
    ("s03_injury", ["s01_timestamps", "s02_technique"], []),
    ("s04_comparison", ["s01_timestamps", "s02_technique", "s03_injury"], []),
    ("s05_speed", ["s02_technique"], []),
    ("s06_training", ["s04_comparison", "s03_injury"], []),
    ("s07_strength", ["s04_comparison", "s03_injury"], []),
]
SECTION_OF = {"s01_timestamps": "s01", "s02_technique": "s02", "s03_injury": "s03", "s04_comparison": "s04",
              "s05_speed": "s05", "s06_training": "s06", "s07_strength": "s07", "s09_limitations": "s09"}


# ------------------------------------------------------------------ evidence selection
def evidence_lines(evidence: dict, keys: list[str], max_lines: int = 90) -> str:
    """The EVIDENCE block, capped. Keys arrive in priority order, so the cap drops the least
    relevant."""
    seen, lines = set(), []
    for k in keys:
        if k in evidence and k not in seen:
            seen.add(k)
            e = evidence[k]
            conf = f" [{e['confidence']}]" if e.get("confidence") else ""
            lines.append(f"{{{{{k}}}}} = {format_value(e)}{conf}")
        if len(lines) >= max_lines:
            break
    if len(lines) >= max_lines:
        lines.append(f"(evidence list capped at {max_lines} entries; the most relevant are shown)")
    return "\n".join(lines) if lines else "(no measured evidence available for this section)"


def _pref(ev: dict, *prefixes: str) -> list[str]:
    return [k for k in ev if k.startswith(prefixes)]


def quality_keys(metrics: dict) -> list[str]:
    return _pref(metrics["evidence_index"], "session.", "quality.")


def event_keys(metrics: dict) -> list[str]:
    ev = metrics["evidence_index"]
    out = []
    for e in metrics["events"]:
        out += [k for k in (f"event{e['id']}.start_t_s", f"event{e['id']}.end_t_s", f"event{e['id']}.value",
                            f"event{e['id']}.threshold", f"event{e['id']}.n_strides") if k in ev]
    return out


def section_base_keys(sid: str, metrics: dict) -> list[str]:
    ev = metrics["evidence_index"]
    tech = [k for row in metrics["technique"].values() for k in row["keys"]]
    if sid == "s09_limitations":
        return quality_keys(metrics) + ["screen.airborne_min_frames"]
    if sid == "s01_timestamps":
        return event_keys(metrics) + [k for k in ev if k.startswith("screen.")] + ["events.count"]
    if sid == "s02_technique":
        knee = [k for k in ev if k.startswith(("L.LOADING.knee_deg", "R.LOADING.knee_deg", "L.MID_STANCE.knee_deg", "R.MID_STANCE.knee_deg"))
                and k.endswith((".min", ".mean"))]
        return tech + knee + [k for k in ev if k.startswith("both.stride.") and k.endswith((".mean", ".sd"))][:20] + quality_keys(metrics)
    if sid == "s03_injury":
        return (_pref(ev, "risk.") + _pref(ev, "asym.") + _pref(ev, "drift.") + _pref(ev, "early.", "late.")
                + ["gait.straight_leg_flag_share_pct"] + event_keys(metrics))
    if sid == "s04_comparison":
        return (event_keys(metrics) + tech + _pref(ev, "asym.") + _pref(ev, "drift.")
                + [k for k in ev if k.startswith("both.stride.") and k.endswith(".mean")][:14])
    if sid == "s05_speed":
        return (_pref(ev, "gait.") + ["session.speed_kmh", "both.stride.step_length_leg.mean", "asym.step_time.pct",
                                      "both.stride.vertical_osc_leg.mean", "both.stride.stride_time_s.mean"])
    return quality_keys(metrics)[:6]


def _shrink(val, max_chars: int):
    """Earlier sections are passed on for consistency, not re-analysis: keep ids, enums and
    headline text, drop evidence_keys and clip long prose."""
    if isinstance(val, dict):
        return {k: _shrink(v, max_chars) for k, v in val.items() if k != "evidence_keys"}
    if isinstance(val, list):
        return [_shrink(v, max_chars) for v in val]
    if isinstance(val, str) and len(val) > max_chars:
        return val[:max_chars].rsplit(" ", 1)[0] + " ..."
    return val


KEY_RE = re.compile(r"(?:L|R|both|all|session|quality|benchmark|gait|asym|drift|early|late|risk|screen|events|event\d+)\.[\w.\-]+")


def digest(outputs: dict, deps: list[str], max_items: int = 4, max_chars: int = 220,
           max_section_chars: int = 2200) -> tuple[str, list[str]]:
    """Earlier sections as compact JSON, plus every evidence key they cite."""
    parts, keys = [], []
    for d in deps:
        val = outputs.get(d)
        if val is None:
            continue
        val = _shrink(val, max_chars)
        blob = json.dumps(val, separators=(",", ":"), ensure_ascii=False)
        if len(blob) > max_section_chars:
            blob = blob[:max_section_chars] + ' ..."(truncated)"}'
        parts.append(f"### {d}\n{blob}")
        for _, s in walk_strings(val):
            keys += keys_in(s)
            if KEY_RE.fullmatch(s):
                keys.append(s)
    return "\n\n".join(parts), keys


def context_brief(metrics: dict, outputs: dict) -> dict:
    session = metrics["session"]
    a = session["athlete"]
    events = [{"id": e["id"], "type": e["type"], "name": e["display_name"], "severity": e["severity"],
               "confidence": e["confidence"], "start_key": f"event{e['id']}.start_t_s"} for e in metrics["events"]]
    brief = {
        "athlete": {k: a.get(k) for k in ("athlete_name", "level", "discipline", "setting", "camera_view")},
        "camera_view_class": session["view_class"],
        "strides_analysed": session["n_strides"],
        "overall_analysis_confidence": metrics["data_quality"]["overall_analysis_confidence"],
        "camera_confidence_caps": metrics["data_quality"]["camera_confidence_cap"],
        "speed_provided": session.get("speed_kmh") is not None,
        "events": events,
        "technique_status": {p: {"status": metrics["technique"][TECH_KEY[p]]["status"],
                                 "confidence_ceiling": metrics["technique"][TECH_KEY[p]]["confidence"]}
                             for p in PARAMETERS},
        "injury_levels": {c: {"label": r["label"], "level": r["level"], "confidence_ceiling": r.get("confidence")}
                          for c, r in metrics["risk"].items()},
        "drift_status": metrics.get("drift_status") or "available",
    }
    ids = valid_ids(outputs)
    if any(ids.values()):
        brief["valid_ids"] = ids
    return brief


def build_prompt(sid: str, cfg_prompts: dict, metrics: dict, outputs: dict, deps: list[str],
                 has_images: bool, limits: dict | None = None) -> tuple[str, str, dict]:
    limits = limits or {}
    ev = metrics["evidence_index"]
    ctx = {"ids": valid_ids(outputs), "event_ids": [e["id"] for e in metrics["events"]]}
    ctx["timestamp_keys"] = [f"event{i}.start_t_s" for i in ctx["event_ids"] if f"event{i}.start_t_s" in ev]
    prior, prior_keys = digest(outputs, deps, max_items=int(limits.get("max_prior_points", 4)),
                               max_chars=int(limits.get("max_prior_chars", 220)),
                               max_section_chars=int(limits.get("max_prior_section_chars", 2200)))
    cap_lines = int(limits.get("max_evidence_lines", 90))
    call_keys = [k for k in dict.fromkeys(section_base_keys(sid, metrics) + prior_keys) if k in ev][:cap_lines]
    ctx["evidence_keys"] = call_keys
    rules = cfg_prompts["sections"][sid]
    system = cfg_prompts["system"]
    header = [f"SECTION: {sid}"]
    brief = context_brief(metrics, outputs)
    user = "\n\n".join(filter(None, [
        "\n".join(header),
        "INSTRUCTIONS\n" + rules.strip(),
        "CONTEXT\n" + json.dumps(brief, ensure_ascii=False),
        (f"EARLIER SECTIONS (cite their ids; do not contradict them)\n{prior}") if prior else "",
        "EVIDENCE (cite values ONLY as {{key}})\n" + evidence_lines(ev, call_keys, cap_lines),
        ("IMAGE: the attached annotated frame(s) may be used only for OBSERVED qualitative points "
         "(arm carriage, head position, visible tension, occlusion, clothing). Numbers printed on the image "
         "must still be cited by key." if has_images else
         "NO IMAGE is available: anything that needs visual inspection is NOT RELIABLY ASSESSABLE FROM "
         "AVAILABLE VIDEO."),
    ]))
    schema = schema_for(sid, ctx)
    user += "\n\n" + output_format(schema)
    return system, user, schema


# ------------------------------------------------------------------ self-checks
def _ids(items, key="id"):
    return [i.get(key) for i in items]


def local_checks(sid: str, out: dict, metrics: dict, outputs: dict, banned: list[str] | None = None) -> list[str]:
    """Rules one section must satisfy on its own. Run by S8 before accepting an output, and
    again by S9."""
    v: list[str] = []

    def need_keys(items, label):
        for i, it in enumerate(items):
            if not it.get("evidence_keys"):
                v.append(f"{label}[{i}] has no evidence_keys")

    def presc(obj_, label):
        for path, text in walk_strings(obj_):
            leaf = re.sub(r"\[\d+\]$", "", path.split(".")[-1])
            if leaf in PRESCRIPTIVE and MEASUREMENT_LIKE.search(text):
                v.append(f"{label}.{path}: a measurement-like number (degrees, percent, cadence, speed) "
                         f"appears in a training field. Cite an evidence KEY or drop it.")

    events = metrics["events"]
    session = metrics["session"]
    if sid == "s01_timestamps":
        got = sorted(e["event_id"] for e in out["events"])
        want = sorted(e["id"] for e in events)
        if got != want:
            v.append(f"events must cover exactly {want}, each once; got {got}")
        need_keys(out["events"], "events")
    elif sid == "s02_technique":
        if sorted(r["parameter"] for r in out["rows"]) != sorted(PARAMETERS):
            v.append("each of the six parameters exactly once")
        for r in out["rows"]:
            t = metrics["technique"][TECH_KEY[r["parameter"]]]
            cap = t.get("confidence")
            if cap and CONF.index(r["confidence"]) < CONF.index(cap):
                v.append(f"{r['parameter']}: confidence {r['confidence']} exceeds the camera-view ceiling {cap}")
            if t["status"].startswith(NA) or NA in t["status"]:
                if NA not in r["observation"]:
                    v.append(f"{r['parameter']} is not assessable from this view; the observation must say {NA}")
            elif not r["evidence_keys"]:
                v.append(f"{r['parameter']} has no evidence_keys")
    elif sid == "s03_injury":
        if sorted(r["category"] for r in out["risks"]) != sorted(RISK_CATS):
            v.append("each of the five categories exactly once")
        for r in out["risks"]:
            lvl = metrics["risk"][r["category"]]
            cap = lvl.get("confidence")
            if lvl["level"] == "NOT ASSESSED":
                if NA not in r["rationale"]:
                    v.append(f"{r['category']} is NOT ASSESSED; the rationale must say {NA}")
            else:
                if not r["evidence_keys"]:
                    v.append(f"{r['category']} has no evidence_keys")
                if cap and CONF.index(r["confidence"]) < CONF.index(cap):
                    v.append(f"{r['category']}: confidence {r['confidence']} exceeds the ceiling {cap}")
        for path, text in walk_strings(out):
            low = text.lower()
            for b in banned or []:
                if re.search(rf"\b{re.escape(b)}\b", low):
                    v.append(f"{path}: diagnostic language '{b}' is not allowed")
    elif sid == "s04_comparison":
        S, Wk = out["strengths"], out["weaknesses"]
        for label, items, pat in (("strength", S, r"S\d+"), ("weakness", Wk, r"W\d+")):
            ids = _ids(items)
            if len(set(ids)) != len(ids) or any(not re.fullmatch(pat, i or "") for i in ids):
                v.append(f"{label} ids must be unique; got {ids}")
        ranks = sorted(w["rank"] for w in Wk)
        if ranks != list(range(1, len(ranks) + 1)):
            v.append(f"weakness ranks must be 1..n with no gaps, got {ranks}")
        need_keys(S, "strengths")
        need_keys(Wk, "weaknesses")
        serious = [e for e in events if e["severity"] in ("MODERATE", "CRITICAL")]
        cited = {w["timestamp_key"] for w in Wk}
        if serious and not any(f"event{e['id']}.start_t_s" in cited for e in serious):
            v.append("at least one MODERATE or CRITICAL flagged event must appear as a weakness timestamp_key")
    elif sid == "s05_speed":
        if sorted(r["metric"] for r in out["rows"]) != sorted(SPEED_ROWS):
            v.append("each of the five metrics exactly once")
        if session.get("speed_kmh") is None:
            for r in out["rows"]:
                if r["metric"] == "Pace" and "NOT PROVIDED" not in (r["reading"] + r["note"]) and NA not in (r["reading"] + r["note"]):
                    v.append("speed was NOT PROVIDED; Pace must say NOT PROVIDED or NOT RELIABLY ASSESSABLE")
    elif sid == "s06_training":
        W = set(valid_ids(outputs)["weaknesses"])
        ranks = sorted(i["rank"] for i in out["immediate"])
        if ranks != list(range(1, len(ranks) + 1)):
            v.append(f"immediate corrections ranked 1..n, got {ranks}")
        for i in out["immediate"]:
            if i["issue_ref"] not in W and i["issue_ref"] != NOT_ESTABLISHED:
                v.append(f"immediate rank {i['rank']} references unknown id {i['issue_ref']}")
        top = [w["id"] for w in sorted(outputs.get("s04_comparison", {}).get("weaknesses", []), key=lambda w: w["rank"])[:2]]
        missing = [t for t in top if t not in {i["issue_ref"] for i in out["immediate"]}]
        if missing:
            v.append(f"the top-ranked weaknesses {missing} must each have an immediate correction")
        ev_ = metrics["evidence_index"]
        for m in out["milestones"]:
            if m["metric_key"] not in ev_:
                v.append(f"milestone metric_key '{m['metric_key']}' is not an evidence key")
        presc(out, "s06")
    elif sid == "s07_strength":
        W = set(valid_ids(outputs)["weaknesses"])
        for i, e in enumerate(out["exercises"]):
            bad = [a for a in e["addresses"] if a not in W and a != NOT_ESTABLISHED]
            if bad:
                v.append(f"exercises[{i}] addresses unknown ids {bad}")
        presc(out, "s07")
    return v
