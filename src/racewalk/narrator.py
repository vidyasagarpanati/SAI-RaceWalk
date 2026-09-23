"""Narration engine shared by S8 (write) and S9 (repair).

Each section is generated, then checked by the agent itself before it is
accepted: JSON schema, evidence grounding, and the section's own rules. A
rejected output goes back to the model once more with the exact violations
listed. Temperature is 0, so the feedback is what changes the answer.
"""
from __future__ import annotations

import json
import re
import time as _time
from pathlib import Path

import cv2
import jsonschema

from racewalk import grounding
from racewalk.io_guard import guarded_open, guarded_path
from racewalk.report_spec import (CALLS, PRESCRIPTIVE, SKIP_FIELDS, build_prompt, local_checks,
                                 schema_for, valid_ids)

EVIDENCE_LINE = re.compile(r"^\{\{([\w.\-]+)\}\} = ", re.M)

# Failure classes, reported separately so a run says WHAT kind of problem it hit.
TRANSPORT = "TRANSPORT"      # unreachable, timeout, truncated or unparseable reply
SCHEMA = "SCHEMA"            # valid JSON, wrong shape
GROUNDING = "GROUNDING"      # typed number, unknown evidence key
RULE = "RULE"                # section rule (ids, counts, coverage, banned language)


def classify(problem: str) -> str:
    if problem.startswith("schema:"):
        return SCHEMA
    if "was Model reply" in problem or "cut off" in problem:
        return TRANSPORT
    if "evidence" in problem or "typed outside" in problem or "braces" in problem:
        return GROUNDING
    return RULE


ESCALATION = {
    1: "This is attempt two. Fix ONLY the problems listed and change nothing else.",
    2: "This is attempt three. Be brief. Remove any sentence you cannot express with a "
       "{{KEY}}; a shorter, correct section is better than a longer one.",
    3: "FINAL attempt. Return the smallest valid answer that satisfies the schema and the "
       "listed problems. Use NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO wherever you "
       "cannot cite evidence.",
}
NUMBER_HINT = ("Remove every typed number that is not a {{KEY}}: no thresholds, targets, "
               "approximations or restated values. Cite the {{KEY}} instead, or drop the number.")


def _normalise_keys(obj):
    """Strip braces/backticks a model may put around evidence_keys entries."""
    if isinstance(obj, dict):
        return {k: ([re.sub(r"[{}`\s]", "", x) if isinstance(x, str) else x for x in v]
                    if k == "evidence_keys" and isinstance(v, list) else _normalise_keys(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise_keys(x) for x in obj]
    return obj


class Narrator:
    def __init__(self, ctx, llm=None):
        self.ctx = ctx
        self.metrics = ctx.read_json("05_metrics.json")
        self.manifest = ctx.read_json("06_manifest.json")
        self.ev = self.metrics["evidence_index"]
        self.dir = guarded_path(ctx.narrative_dir)
        (self.dir / "cache").mkdir(parents=True, exist_ok=True)
        if llm is None:
            from racewalk.llm import OllamaClient
            llm = OllamaClient(ctx.cfg, self.dir / "cache")
        self.llm = llm
        self.max_retries = int(ctx.cfg.get("llm.max_retries_per_section", 2))
        self.limits = {k: ctx.cfg.get(f"llm.{k}") for k in
                       ("max_evidence_lines", "max_prior_points", "max_prior_chars")}
        self.limits = {k: v for k, v in self.limits.items() if v is not None}
        self.warn_tokens = int(ctx.cfg.get("llm.warn_prompt_tokens", 7000))
        self.call_seconds: list[float] = []
        self.total_calls = 0
        self.done_calls = 0
        caps = llm.capabilities()
        self.vision = "vision" in caps and bool(ctx.cfg.get("llm.send_key_frames_to_model", True))
        banned_file = ctx.cfg.root / "config" / "banned_phrases.txt"
        self.banned = [ln.strip().lower() for ln in banned_file.read_text(encoding="utf-8").splitlines()
                       if ln.strip() and not ln.startswith("#")] if banned_file.is_file() else []
        self.outputs: dict = {}
        self.log: list[dict] = []
        self.links: list[str] = []
        self.status: dict[str, dict] = {}
        self.backoff_base = float(ctx.cfg.get("llm.retry_backoff_base_s", 1.0))
        self.backoff_max = float(ctx.cfg.get("llm.retry_backoff_max_s", 30.0))
        self._frame_w = None

    # ------------------------------------------------------------ helpers
    def _images(self, kinds: list[str]) -> list[bytes]:
        """Annotated key frames for a call: athlete area only (no side panel), downscaled.
        'event' picks the most severe flagged-event frames, 'reference' the mid-stance
        frame of each reference stride."""
        if not self.vision or not kinds:
            return []
        max_w = int(self.ctx.cfg.get("report.model_image_max_width", 768))
        limit = int(self.ctx.cfg.get("llm.max_images_per_section", 2))
        frames = self.manifest["frames"]
        chosen: list[dict] = []
        sev = {"CRITICAL": 0, "MODERATE": 1, "MINOR": 2}
        by_id = {e["id"]: e for e in self.metrics["events"]}
        for kind in kinds:
            if kind == "event":
                pool = [f for f in frames if f["kind"] == "event"]
                pool.sort(key=lambda f: (sev[by_id[f["event_id"]]["severity"]], f["event_id"]))
            else:
                pool = [f for f in frames if f["kind"] == "reference" and f["phase"] == "MID_STANCE"]
            chosen += pool
        out = []
        for f in chosen[:limit]:
            img = cv2.imread(f["file"])
            if img is None:
                continue
            img = img[:, :img.shape[1] - max(int(0.46 * img.shape[0]), 380)]     # athlete only, no side panel
            if img.shape[1] > max_w:
                img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append(buf.tobytes())
        return out

    def digest_view(self) -> dict:
        return dict(self.outputs)

    def schema(self, sid: str) -> dict:
        ev = self.ev
        ids = [e["id"] for e in self.metrics["events"]]
        return schema_for(sid, {"ids": valid_ids(self.outputs), "event_ids": ids,
                                "timestamp_keys": [f"event{i}.start_t_s" for i in ids if f"event{i}.start_t_s" in ev]})

    def check(self, sid: str, out: dict, schema: dict) -> list[str]:
        problems = []
        try:
            jsonschema.validate(out, schema)
        except jsonschema.ValidationError as exc:
            where = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            problems.append(f"schema: {where}: {exc.message[:200]}")
            return problems
        problems += grounding.check_object(out, self.ev, PRESCRIPTIVE, SKIP_FIELDS)
        problems += local_checks(sid, out, self.metrics, self.outputs, self.banned)
        return problems

    # ------------------------------------------------------------ generation
    def generate(self, sid: str, deps: list[str], img_kinds: list[str],
                 extra_feedback: list[str] | None = None) -> tuple[dict, list[str], int]:
        feedback = list(extra_feedback or [])
        images = self._images(img_kinds)
        label = sid
        out, problems = {}, ["not generated"]
        for attempt in range(self.max_retries + 1):
            system, user, schema = build_prompt(
                sid, self.ctx.cfg.prompts, self.metrics, self.digest_view(), deps,
                bool(images), self.limits)
            if feedback:
                user += (f"\n\nYOUR PREVIOUS ANSWER WAS REJECTED (attempt {attempt} of "
                         f"{self.max_retries + 1}). {ESCALATION.get(attempt, '')}\n"
                         f"Fix exactly these problems:\n- " + "\n- ".join(feedback[:15]))
                delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
                if delay:
                    _time.sleep(delay)
            est = len(system) + len(user)
            self.done_calls += 1
            head = (f"  S8 [{self.done_calls}/{self.total_calls or '?'}] {label}"
                    f"{f' retry {attempt}' if attempt else ''} "
                    f"prompt~{est // 4} tok{f', {len(images)} image(s)' if images else ''}")
            print(head + " ...", flush=True)
            if est // 4 > self.warn_tokens:
                print(f"      warning: prompt is larger than llm.warn_prompt_tokens "
                      f"({self.warn_tokens}); lower llm.max_evidence_lines", flush=True)
            started = _time.time()
            try:
                out = self.llm.chat_json(system, user, schema, images, attempt=attempt)
            except Exception as exc:  # noqa: BLE001  (LLMBadOutput and transport errors)
                from racewalk.llm import LLMBadOutput, LLMError
                if not isinstance(exc, (LLMBadOutput, LLMError)):
                    raise
                problems = [f"your answer was {exc}. Answer again with the SAME JSON structure "
                            f"but shorter text fields and fewer list items, and close every "
                            f"bracket and quote."]
                print(f"      rejected: reply was cut off or unparseable; asking again", flush=True)
                self.log.append({"section": sid, "attempt": attempt,
                                 "problems": problems, "images": len(images),
                                 "prompt_chars": est, "seconds": round(_time.time() - started, 1),
                                 "done_reason": exc.done_reason})
                feedback = problems
                out = {}
                continue
            # Link restated evidence values back to their keys (unambiguous only).
            subset = {k: self.ev[k] for k in EVIDENCE_LINE.findall(user) if k in self.ev}
            stats = dict(getattr(self.llm, "last", {}) or {})
            took = stats.get("seconds") or round(_time.time() - started, 1)
            if not stats.get("cached"):
                self.call_seconds.append(took)
            print(f"      {'cached' if stats.get('cached') else f'{took:.0f}s'}, "
                  f"{stats.get('prompt_tokens', 0)}+{stats.get('completion_tokens', 0)} tokens"
                  f"{self._eta()}", flush=True)
            out = _normalise_keys(out)
            out, linked = grounding.link_object(out, subset, self.ev, SKIP_FIELDS, PRESCRIPTIVE)
            self.links += [f"{sid}: {x}" for x in linked]
            problems = self.check(sid, out, schema)
            if problems:
                print(f"      rejected: {problems[0][:120]}", flush=True)
            self.log.append({"section": sid, "attempt": attempt,
                             "auto_linked": linked[:30], "problems": problems[:20],
                             "images": len(images), "prompt_chars": est,
                             "prompt_tokens": stats.get("prompt_tokens"),
                             "completion_tokens": stats.get("completion_tokens"),
                             "seconds": took, "cached": bool(stats.get("cached"))})
            hints = [NUMBER_HINT] if any("typed outside" in p for p in problems) else []
            if not problems:
                return out, [], attempt
            feedback = problems + hints
        return out, problems, self.max_retries

    def _record_failure(self, key: str, problems: list[str], attempts: int) -> None:
        self.status[key] = {"status": "FAILED", "attempts": attempts + 1,
                            "classes": sorted({classify(p) for p in problems}),
                            "problems": problems[:10]}

    def _eta(self) -> str:
        if not self.call_seconds or not self.total_calls:
            return ""
        typical = sorted(self.call_seconds)[len(self.call_seconds) // 2]
        left = max(0, self.total_calls - self.done_calls)
        if not left:
            return ""
        secs = typical * left
        return f", about {secs / 60:.0f} min left ({left} call(s))" if secs >= 90 else f", ~{secs:.0f}s left"

    def save(self, name: str, payload) -> Path:
        path = self.dir / f"{name}.json"
        with guarded_open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path

    def run_all(self) -> dict:
        failed: dict[str, list[str]] = {}
        retries = 0
        self.total_calls = len(CALLS)
        started_all = _time.time()
        print(f"  S8: {self.total_calls} model call(s) planned, model {self.ctx.cfg.get('llm.model')}", flush=True)
        for sid, deps, img_kinds in CALLS:
            out, probs, att = self.generate(sid, deps, img_kinds)
            retries += att
            if probs:
                failed[sid] = probs
                self._record_failure(sid, probs, att)
                self.save(sid, {"_status": "FAILED", "_problems": probs})
                continue          # the section is left out; dependants degrade gracefully
            self.outputs[sid] = out
            self.status[sid] = {"status": "OK", "attempts": att + 1}
            self.save(sid, out)
        self.save("all_sections", self.outputs)
        self.save("section_status", self.status)
        self.save("generation_log", self.log)
        self.save("auto_links", self.links)
        usage = dict(getattr(self.llm, "usage", {}))
        usage["vision_used"] = self.vision
        usage["wall_seconds"] = round(_time.time() - started_all, 1)
        usage["median_call_seconds"] = (sorted(self.call_seconds)[len(self.call_seconds) // 2]
                                        if self.call_seconds else 0)
        print(f"  S8 finished in {usage['wall_seconds'] / 60:.1f} min "
              f"({usage['calls']} calls, {usage['cached']} cached, "
              f"{usage['prompt_tokens']}+{usage['completion_tokens']} tokens)", flush=True)
        self.save("usage", usage)
        classes: dict[str, int] = {}
        for probs in failed.values():
            for p in probs:
                classes[classify(p)] = classes.get(classify(p), 0) + 1
        if failed:
            print(f"  S8: {len(failed)} section(s) could not be produced after "
                  f"{self.max_retries + 1} attempts: {', '.join(sorted(failed))}", flush=True)
            print(f"      failure classes: {classes}", flush=True)
        return {"failed": failed, "retries": retries, "usage": usage,
                "auto_linked": len(self.links), "classes": classes,
                "status": self.status}

    def load_saved(self) -> None:
        self.outputs = json.loads((self.dir / "all_sections.json").read_text(encoding="utf-8"))
