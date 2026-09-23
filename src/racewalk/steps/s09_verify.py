"""S9 verify: the guardrail agent.

IN     05_metrics.json, 08_narrative/all_sections.json
DO     the consistency checklist, executed mechanically, plus evidence grounding of every
       sentence and cross-section consistency. Sections that fail are sent back to the
       narrator with the violations, up to llm.max_retries_per_section rounds, then
       re-verified.
OUT    09_verification.json (checklist, per-section results, repairs made)
VERIFY every checklist item PASS. A failure blocks a COMPLETE report: the report is either
       published PARTIAL with the failing sections marked NOT AVAILABLE, or not at all.
"""
from __future__ import annotations

import json as _json

from racewalk.context import Context
from racewalk.contracts import StepResult
from racewalk.narrator import Narrator
from racewalk.report_spec import CALLS, MASTER, PARAMETERS, report_order, valid_ids


def _section_problems(nar: Narrator) -> dict[str, list[str]]:
    probs: dict[str, list[str]] = {}
    for sid, _, _ in CALLS:
        out = nar.outputs.get(sid)
        if out is None:
            probs[sid] = ["section missing"]
            continue
        p = nar.check(sid, out, nar.schema(sid))
        if p:
            probs[sid] = p
    return probs


def _cross_problems(nar: Narrator) -> dict[str, list[str]]:
    o = nar.outputs
    probs: dict[str, list[str]] = {}
    W = sorted(o.get("s04_comparison", {}).get("weaknesses", []), key=lambda w: w["rank"])
    top = W[0]["id"] if W else None
    imm = sorted(o.get("s06_training", {}).get("immediate", []), key=lambda i: i["rank"])
    if top and imm and imm[0]["issue_ref"] != top:
        probs.setdefault("s06_training", []).append(
            f"immediate correction rank 1 must address the top-ranked weakness {top}, not {imm[0]['issue_ref']}")
    st = o.get("s07_strength", {}).get("exercises", [])
    if top and st and not any(top in e["addresses"] for e in st):
        probs.setdefault("s07_strength", []).append(f"at least one exercise must address the top-ranked weakness {top}")
    return probs


def _checklist(nar: Narrator, sec: dict, cross: dict) -> list[dict]:
    m, o = nar.metrics, nar.outputs
    order_ok = [s["key"] for s in report_order()] == [k for k, _ in MASTER]
    grounded = not any("evidence key" in p or "typed outside" in p for ps in sec.values() for p in ps)
    no_schema = not any(p.startswith("schema") for ps in sec.values() for p in ps)
    all_present = all(s in o for s, _, _ in CALLS)
    injury_ok = not any("diagnostic language" in p for p in sec.get("s03_injury", []))
    ceiling_ok = not any("ceiling" in p for ps in sec.values() for p in ps)
    train_ok = not any("measurement-like" in p for ps in sec.values() for p in ps)
    rows = [
        ("All required sections are present.", all_present),
        ("Section order is unchanged.", order_ok),
        ("Every flagged event is described exactly once.", "s01_timestamps" in o and "s01_timestamps" not in sec),
        ("All six technique parameters are covered.",
         sorted(r["parameter"] for r in o.get("s02_technique", {}).get("rows", [])) == sorted(PARAMETERS)),
        ("Measurements are supported by measured data.", grounded),
        ("No measurement has been fabricated.", grounded),
        ("All major findings have confidence ratings.", no_schema),
        ("Confidence never exceeds the camera-view ceiling.", ceiling_ok),
        ("Weaknesses are ranked and every serious flagged event is shown by one.", "s04_comparison" not in sec),
        ("Injury assessment is non-diagnostic.", injury_ok),
        ("Pace is reported only when the belt speed was provided.", "s05_speed" not in sec),
        ("Recommendations correspond to identified weaknesses.",
         "s06_training" not in sec and "s07_strength" not in sec and not cross),
        ("Training fields contain no invented measurement targets.", train_ok),
        ("Unsupported conclusions are marked NOT ASSESSABLE.", grounded and "s03_injury" not in sec),
        ("No required section has been omitted.", all_present),
        ("Benchmarks: only cited entries used.", all(r.get("source") for r in m["benchmarks"]["rows"])),
    ]
    return [{"check": c, "status": "PASS" if ok else "FAIL"} for c, ok in rows]


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S9")
    nar = Narrator(ctx, llm=ctx.llm)
    nar.load_saved()
    status_file = nar.dir / "section_status.json"
    if status_file.is_file():
        nar.status = _json.loads(status_file.read_text(encoding="utf-8"))
    repairs = []
    rounds = int(ctx.cfg.get("llm.max_retries_per_section", 2))
    deps_of = {sid: (deps, imgs) for sid, deps, imgs in CALLS}

    for rnd in range(rounds + 1):
        sec = _section_problems(nar)
        cross = _cross_problems(nar)
        combined: dict[str, list[str]] = {}
        for d in (sec, cross):
            for k, v in d.items():
                combined.setdefault(k, []).extend(v)
        if not combined or rnd == rounds:
            break
        exhausted = {k for k, st in nar.status.items()
                     if st.get("status") == "FAILED" and st.get("attempts", 0) > rounds}
        for key, problems in combined.items():
            # S8 already spent every attempt on these; only cross-section findings, which
            # S8 could not see, are worth another call.
            if key in exhausted and key not in cross:
                continue
            deps, imgs = deps_of[key]
            out, left, attempts = nar.generate(key, deps, imgs, extra_feedback=problems)
            if left:
                # A repair that still fails is NOT accepted: it would put unverified text in
                # the report. The section stays absent and renders as NOT AVAILABLE.
                nar._record_failure(key, left, attempts)
                nar.outputs.pop(key, None)
                nar.save(key, {"_status": "FAILED", "_problems": left})
            else:
                nar.outputs[key] = out
                nar.status[key] = {"status": "OK", "attempts": attempts + 1, "repaired": True}
                nar.save(key, out)
            repairs.append({"round": rnd + 1, "section": key, "problems": problems[:10], "resolved": not left})
        nar.save("all_sections", nar.outputs)
        nar.save("section_status", nar.status)

    checklist = _checklist(nar, sec, cross)
    missing = [sid for sid, _, _ in CALLS if sid not in nar.outputs]
    complete = not combined and all(r["status"] == "PASS" for r in checklist) and not missing
    payload = {"passed": complete, "partial": bool(missing or combined) and bool(nar.outputs),
               "allow_partial": bool(ctx.cfg.get("report.allow_partial", True)),
               "sections_available": sorted(nar.outputs), "missing_sections": missing,
               "checklist": checklist, "section_problems": sec, "cross_section_problems": cross,
               "repairs": repairs}
    out = ctx.write_json("09_verification.json", payload)

    partial_ok = payload["allow_partial"] and bool(nar.outputs)
    sev = "WARN" if partial_ok else "FAIL"
    for row in checklist:
        res.check(row["check"], row["status"] == "PASS",
                  "" if row["status"] == "PASS" else "See 09_verification.json for the offending text.", severity=sev)
    for key, problems in combined.items():
        res.check(f"section_{key}", False, "; ".join(problems[:5]), severity=sev)
    res.check("narrative_available", bool(nar.outputs),
              f"{len(nar.outputs)} section(s) verified." if nar.outputs else
              "Nothing to verify: no section was produced.")
    if missing:
        res.check("all_sections_available", False, f"Rendering PARTIAL: missing {missing}", severity=sev)
    res.outputs["verification"] = str(out)
    res.outputs["narrative"] = str(nar.dir / "all_sections.json")
    res.stats = {"passed": payload["passed"], "partial": payload["partial"], "missing": len(missing),
                 "repairs": len(repairs), "checklist_pass": sum(r["status"] == "PASS" for r in checklist),
                 "checklist_total": len(checklist)}
    return res
