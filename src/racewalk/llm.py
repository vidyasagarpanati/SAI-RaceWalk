"""LLM clients.

OllamaClient  the real one: localhost only, temperature 0, fixed seed, JSON
              schema enforced by Ollama's structured outputs, reasoning tokens
              off, every response cached on disk by a hash of the full request.
FakeLLM       deterministic stand-in for tests. Writes schema-valid sections
              that cite evidence keys, so S8, S9 and S10 can be exercised
              without a GPU or a model.

Token accounting: every real call records prompt and completion token counts,
which S8 totals into 08_narrative/usage.json.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMBadOutput(LLMError):
    """The model answered, but not with complete valid JSON (usually truncated).
    Retryable: the narrator sends it back with a request to be more concise."""
    def __init__(self, msg: str, raw: str = "", done_reason: str | None = None):
        super().__init__(msg)
        self.raw, self.done_reason = raw, done_reason


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


class OllamaClient:
    def __init__(self, cfg, cache_dir: Path):
        self.cfg = cfg
        self.base = cfg.get("llm.base_url", "http://localhost:11434").rstrip("/")
        self.model = cfg.get("llm.model")
        if not self.model:
            raise LLMError("config/pipeline.yaml llm.model is not set. Run `ollama list` and set it.")
        self.cache_dir = cache_dir
        self.timeout = float(cfg.get("llm.request_timeout_s", 300))
        self._caps: set[str] | None = None
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        self.last: dict = {}

    def capabilities(self) -> set[str]:
        if self._caps is None:
            import httpx
            try:
                r = httpx.post(f"{self.base}/api/show", json={"model": self.model}, timeout=30)
                r.raise_for_status()
                self._caps = set(r.json().get("capabilities") or [])
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"Ollama at {self.base} not reachable or model {self.model!r} "
                               f"unknown: {exc}") from exc
        return self._caps

    def chat_json(self, system: str, user: str, schema: dict,
                  images: list[bytes] | None = None, attempt: int = 0) -> dict:
        import httpx
        opts = {"temperature": float(self.cfg.get("llm.temperature", 0.0)),
                "seed": int(self.cfg.get("llm.seed", 0)),
                "top_p": float(self.cfg.get("llm.top_p", 1.0)),
                "num_ctx": int(self.cfg.get("llm.num_ctx", 16384)),
                "num_predict": int(self.cfg.get("llm.num_predict", 4096))}
        think = bool(self.cfg.get("llm.think", False))
        key = hashlib.sha256(json.dumps({
            "model": self.model, "opts": opts, "think": think, "system": system, "user": user,
            "schema": schema, "images": [hashlib.sha256(b).hexdigest() for b in images or []],
            "attempt": attempt,
        }, sort_keys=True).encode()).hexdigest()
        cache = self.cache_dir / f"{key[:32]}.json"
        if cache.is_file():
            self.usage["cached"] += 1
            blob = json.loads(cache.read_text(encoding="utf-8"))
            self.last = {"seconds": 0.0, "prompt_tokens": blob.get("prompt_tokens") or 0,
                         "completion_tokens": blob.get("completion_tokens") or 0, "cached": True}
            return blob["parsed"]

        msg = {"role": "user", "content": user}
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
        payload = {"model": self.model, "stream": False, "format": schema, "options": opts,
                   "messages": [{"role": "system", "content": system}, msg]}
        if not think:
            payload["think"] = False
        t0 = time.time()
        r = httpx.post(f"{self.base}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think", None)       # model or server without thinking support
            r = httpx.post(f"{self.base}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise LLMError(f"Ollama returned {r.status_code}: {r.text[:500]}")
        body = r.json()
        content = body.get("message", {}).get("content", "")
        reason = body.get("done_reason")
        try:
            parsed = _parse_json(content)
        except Exception as exc:  # noqa: BLE001
            self.usage["calls"] += 1
            why = ("cut off at the output limit (num_predict)" if reason == "length"
                   else f"not valid JSON (done_reason={reason})")
            raise LLMBadOutput(f"Model reply {why}; prompt {body.get('prompt_eval_count')} tokens, "
                               f"reply {body.get('eval_count')} tokens. Start: {content[:160]!r}",
                               raw=content, done_reason=reason) from exc
        dt = time.time() - t0
        self.last = {"seconds": round(dt, 1), "prompt_tokens": int(body.get("prompt_eval_count") or 0),
                     "completion_tokens": int(body.get("eval_count") or 0), "cached": False}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(body.get("prompt_eval_count") or 0)
        self.usage["completion_tokens"] += int(body.get("eval_count") or 0)
        self.usage["seconds"] += dt
        from racewalk.io_guard import guarded_open
        with guarded_open(cache, "w", encoding="utf-8") as fh:
            json.dump({"model": self.model, "options": opts, "seconds": round(dt, 1),
                       "prompt_tokens": body.get("prompt_eval_count"),
                       "completion_tokens": body.get("eval_count"),
                       "raw": content, "parsed": parsed}, fh, indent=1)
        return parsed


# ---------------------------------------------------------------- fake for tests
class FakeLLM:
    """Deterministic, schema-valid sections built from the EVIDENCE keys in the prompt.
    ``fabricate_in`` injects a typed number once, to prove S8 catches it."""

    def __init__(self, fabricate_in: str | None = None, vision: bool = True,
                 truncate_in: str | None = None):
        self.fabricate_in = fabricate_in
        self.truncate_in = truncate_in
        self._truncated = False
        self._fabricated = False
        self.vision = vision
        self.calls: list[str] = []
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def capabilities(self) -> set[str]:
        return {"completion", "vision"} if self.vision else {"completion"}

    def chat_json(self, system, user, schema, images=None, attempt: int = 0) -> dict:
        sid = re.search(r"SECTION: (\w+)", user).group(1)
        keys = re.findall(r"^\{\{([\w.\-]+)\}\} = ", user, flags=re.M)
        ctxj = json.loads(re.search(r"CONTEXT\n(\{.*\})", user).group(1))
        prior = re.findall(r'"id":"([SWE]\d+)"', user)
        self.calls.append(sid)
        self.usage["calls"] += 1
        self.last = {"seconds": 0.0, "prompt_tokens": len(user) // 4, "completion_tokens": 200, "cached": False}
        self.last_user, self.last_schema, self.last_images = user, schema, images or []
        if self.truncate_in and sid == self.truncate_in and not self._truncated:
            self._truncated = True
            raise LLMBadOutput("cut off at the output limit (num_predict)",
                               raw='{"rows": [{"parameter": "Contact", "observation": "Both fe', done_reason="length")
        out = self._build(sid, keys, ctxj, prior, schema)
        if self.fabricate_in == sid and not self._fabricated:
            self._fabricated = True
            _inject_number(out)
        return out

    def _build(self, sid, keys, ctxj, prior, schema):
        NA_ = "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO"
        k = lambda i=0: keys[i % len(keys)] if keys else None  # noqa: E731
        cite = lambda i=0: (f"value {{{{{k(i)}}}}}" if k(i) else NA_)  # noqa: E731
        ek = lambda i=0: [k(i)] if k(i) else []  # noqa: E731
        find = lambda tok, default=0: next((x for x in keys if tok in x), k(default))  # noqa: E731
        cmax = lambda ceiling, want="MEDIUM": ("LOW" if ceiling in (None, "LOW") else want)  # noqa: E731
        W = [p for p in prior if p.startswith("W")] or ["W1", "W2"]
        events = ctxj["events"]
        if sid == "s09_limitations":
            return {"camera": "A single camera limits depth, so leg flexion is estimated.",
                    "lighting": "Adequate for landmark tracking.", "occlusion": "Arms partly hide the hips at times.",
                    "motion_blur": "Low at this shutter speed.", "clothing": "Loose shorts may hide the knee line.",
                    "additional": ["Frame rate limits how short a loss of contact can be seen."]}
        if sid == "s01_timestamps":
            return {"summary": "The screen flagged the events listed below." if events else
                    "No event was flagged; contact and straight leg cannot be ruled on from this view.",
                    "events": [{"event_id": e["id"], "what": f"Flagged from {{{{event{e['id']}.start_t_s}}}} to "
                                                              f"{{{{event{e['id']}.end_t_s}}}}, value {{{{event{e['id']}.value}}}}.",
                                "why_it_matters": "Relevant to the contact and straight-leg rules.",
                                "coaching_cue": "Reach long and land tall.", "confidence": cmax(e["confidence"]),
                                "evidence_keys": [f"event{e['id']}.value"]} for e in events]}
        if sid == "s02_technique":
            tok = {"Contact": "flight", "Straight leg": "straight_leg", "Pelvic rotation": "pelvic_yaw",
                   "Torso posture": "trunk_incl", "Stride length and frequency": "cadence", "Foot strike": "heel_first"}
            rows = []
            for i, (param, info) in enumerate(ctxj["technique_status"].items()):
                na = NA_ in info["status"]
                key = find(tok[param], i)
                rows.append({"parameter": param,
                             "observation": NA_ if na else f"Measured {{{{{key}}}}}.",
                             "why_it_matters": "It bears on legal and efficient technique.",
                             "confidence": "LOW" if na else cmax(info["confidence_ceiling"]),
                             "evidence_keys": [] if na else [key]})
            return {"rows": rows}
        if sid == "s03_injury":
            risks = []
            for i, (cat, info) in enumerate(ctxj["injury_levels"].items()):
                na = info["level"] == "NOT ASSESSED"
                key = find(f"risk.{cat}.driver_value", i)
                risks.append({"category": cat,
                              "rationale": NA_ if na else f"The level is driven by {{{{{key}}}}}.",
                              "possible_concern": "Potential movement-related risk factor for load on this region.",
                              "corrective_focus": "Preventive strength and mobility work.",
                              "confidence": "LOW" if na else cmax(info["confidence_ceiling"]),
                              "evidence_keys": [] if na else [key]})
            return {"risks": risks}
        if sid == "s04_comparison":
            serious = [e for e in events if e["severity"] in ("MODERATE", "CRITICAL")]
            ts = (serious or events or [{"start_key": "NOT ESTABLISHED"}])[0]["start_key"]
            ts_enum = schema["properties"]["weaknesses"]["items"]["properties"]["timestamp_key"]["enum"]
            ts = ts if ts in ts_enum else ts_enum[0]
            return {"strengths": [{"id": f"S{i + 1}", "strength": ["Consistent cadence", "Stable trunk"][i],
                                   "evidence": cite(i), "confidence": "MEDIUM", "evidence_keys": ek(i)} for i in range(2)],
                    "weaknesses": [{"id": f"W{i + 1}", "rank": i + 1,
                                    "weakness": ["Knee flexion during loading", "Late-clip form drift"][i],
                                    "evidence": cite(i + 2), "likely_cause": "Reduced hip extension reach",
                                    "correction": "Cue a straighter reach before contact",
                                    "timestamp_key": ts if i == 0 else "NOT ESTABLISHED",
                                    "priority": "MODERATE" if i == 0 else "MINOR", "confidence": "MEDIUM",
                                    "evidence_keys": ek(i + 2)} for i in range(2)]}
        if sid == "s05_speed":
            rows = []
            for i, m in enumerate(["Pace", "Cadence", "Step and stride length", "Vertical oscillation", "Efficiency"]):
                if m == "Pace" and not ctxj["speed_provided"]:
                    reading, ev_ = "NOT PROVIDED", []
                elif m == "Efficiency":
                    reading, ev_ = NA_, []
                else:
                    key = find({"Pace": "pace", "Cadence": "cadence_spm", "Step and stride length": "step_length",
                                "Vertical oscillation": "vertical_osc"}[m], i)
                    reading, ev_ = f"Measured {{{{{key}}}}}.", [key]
                rows.append({"metric": m, "reading": reading, "note": "Descriptive only; no reference is available.",
                             "confidence": "LOW" if not ev_ else "MEDIUM", "evidence_keys": ev_})
            return {"rows": rows, "summary": "Cadence is measured; pace needs the belt speed."}
        if sid == "s06_training":
            doses = ["3 x 40 m", "4 x 2 min", "2 x 10 min"]
            return {"immediate": [{"rank": r, "issue_ref": W[(r - 1) % len(W)] if r < 3 else W[0],
                                   "cue": "Reach long, land tall", "drill": "Wall-lean straight-leg drill",
                                   "sets": doses[r - 1], "frequency": "3 sessions a week",
                                   "success_metric": f"Track {{{{{k(0)}}}}} each month"} for r in range(1, 4)],
                    "phases": [{"phase": ph, "focus": "Straight-leg reach and pelvic rotation",
                                "sessions": [{"day": d, "content": f"Drill work then {doses[j]} at controlled pace"}
                                             for j, d in enumerate(["Session A", "Session B", "Session C"])]}
                               for ph in ["Weeks 1-4", "Weeks 5-8", "Weeks 9-12"]],
                    "milestones": [{"when": ph, "metric_key": k(0) or "session.n_strides",
                                    "target": "Move toward straighter loading than the baseline value",
                                    "tracked_by": "Re-film from the same angle every four weeks"}
                                   for ph in ["Weeks 1-4", "Weeks 5-8", "Weeks 9-12"]]}
        if sid == "s07_strength":
            return {"exercises": [{"area": a, "exercise": e, "dose": "3 x 12", "purpose": f"Supports {W[0]}",
                                   "addresses": [W[0]]} for a, e in
                                  [("Hip extensors", "Hip thrust"), ("Quadriceps", "Slow step-down"),
                                   ("Core", "Dead bug"), ("Calves", "Single-leg calf raise")]],
                    "notes": "Progress load only when the movement stays controlled."}
        raise KeyError(sid)


def _inject_number(out: Any) -> None:
    """Replace the first prose string with one containing a typed measurement."""
    def visit(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and len(v) > 12 and k not in ("id", "ref", "phase", "issue_ref", "event_id", "category", "parameter", "metric"):
                    o[k] = v + " The knee sat at 172.4 degrees."
                    return True
                if visit(v):
                    return True
        if isinstance(o, list):
            return any(visit(x) for x in o)
        return False
    visit(out)
