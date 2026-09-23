"""S8 narrate (the only step that uses the language model).

IN     05_metrics.json (sliced per section), 06_manifest.json (key frames)
DO     one Ollama call per report section (one per section
       ), temperature 0, fixed seed, schema-enforced JSON, cached by request
       hash. Each output is self-checked (schema, evidence grounding, section
       rules) and regenerated with the violations listed when it fails.
OUT    08_narrative/<section>.json, all_sections.json, generation_log.json,
       usage.json (token counts)
VERIFY every section generated and passing its own checks
"""
from __future__ import annotations

from racewalk.context import Context
from racewalk.contracts import WARN, StepResult
from racewalk.narrator import Narrator


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S8")
    nar = Narrator(ctx, llm=ctx.llm)
    summary = nar.run_all()
    usage = summary["usage"]

    res.check("vision_available", nar.vision,
              "Key frames sent to the model." if nar.vision else
              "Model lacks vision (or disabled): image-dependent observations are marked "
              "NOT RELIABLY ASSESSABLE.", severity=WARN)
    allow_partial = bool(ctx.cfg.get("report.allow_partial", True))
    failed = summary["failed"]
    res.check("all_sections_pass_self_checks", not failed,
              "All sections passed schema, grounding and section rules." if not failed else
              f"{len(failed)} section(s) failed after {nar.max_retries + 1} attempts "
              f"({summary['classes']}). They will render as NOT AVAILABLE in a PARTIAL report: "
              + "; ".join(f"{k}: {v[0][:110]}" for k, v in list(failed.items())[:4]),
              severity=WARN if allow_partial else "FAIL")
    res.check("narrative_produced", bool(nar.outputs),
              f"{len(nar.outputs)} section(s) available for the report." if nar.outputs else
              "No section could be produced; there is nothing to report.")
    res.check("restated_values_linked", summary["auto_linked"] == 0,
              f"{summary['auto_linked']} restated evidence value(s) were linked back to their keys "
              f"automatically; see 08_narrative/auto_links.json.", severity=WARN)
    res.check("few_retries", summary["retries"] <= 3,
              f"{summary['retries']} regeneration(s) were needed.", severity=WARN)
    res.outputs["narrative"] = str(nar.dir / "all_sections.json")
    res.outputs["usage"] = str(nar.dir / "usage.json")
    res.stats = {"sections": len(nar.outputs), "calls": usage.get("calls"),
                 "cached": usage.get("cached"), "prompt_tokens": usage.get("prompt_tokens"),
                 "completion_tokens": usage.get("completion_tokens"),
                 "retries": summary["retries"], "auto_linked": summary["auto_linked"],
                 "failed_sections": sorted(summary["failed"]), "failure_classes": summary["classes"],
                 "minutes": round(usage.get("wall_seconds", 0) / 60, 1)}
    return res
