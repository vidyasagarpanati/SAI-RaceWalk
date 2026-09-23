"""Evidence grounding: the mechanism that makes a fabricated number impossible.

The model never types a measured number. It writes evidence keys in double
braces, {{L.LOADING.knee_deg.min}}, and the renderer substitutes the value
and unit from 05_metrics.json. This module:

  * finds every placeholder and rejects keys that are not in the evidence index;
  * finds every number typed outside a placeholder and rejects it unless it is
    an allowed reference (Section 4, Phase 5, Priority #2, Stride 1, a 0-10 scale
    mention) or, in prescriptive fields, a training dose (3 sets, 2-4 weeks);
  * substitutes placeholders with formatted values for rendering.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")
# Anything in braces, valid key or not. Needed so a malformed citation such as
# {{s15_priorities.priorities[0].sets}} is caught here rather than surviving
# into the rendered report as literal text.
ANY_BRACED = re.compile(r"\{\{([^{}]*)\}\}")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
REFERENCE_BEFORE = re.compile(
    r"(Section|Sections|Phase|Phases|Priority|Priorities|Stride|Strides|stride|strides|Event|Events|event|events|"
    r"Top|#|Appendix|Rule|Rules|Article|PRIORITY|PHASE|SECTION|STRIDE|EVENT|RULE|[Bb]eat|[Ss]tep|[Cc]ount|"
    r"[Rr]ound|[Ss]tage|[Ww]eeks?|[Dd]ays?|[Ss]ets?|[Ll]evel|[Oo]ption|[Dd]rill|[Zz]one|RPE|Session|Sessions)\s*#?\s*$")
SCALE = re.compile(r"\b0\s*[-–]\s*10\b|/\s*10\b")
DOSE_AFTER = re.compile(
    r"^\s*(?:[-–]\s*\d+\s*)?(?:x|×|sets?|reps?|repetitions?|times?|sessions?|days?|weeks?|months?|"
    r"minutes?|mins?|seconds?|secs?|s\b|per\b|km|bpm|m\b|metres?|meters?|kg|lbs?|%)", re.I)


def keys_in(text: str) -> list[str]:
    return PLACEHOLDER.findall(text or "")


def check_text(text: str, evidence: dict, *, prescriptive: bool = False) -> list[str]:
    """Violations for one field.

    ``prescriptive`` fields (sets, repetitions, frequency, drills) exist to hold
    training doses, so numbers are not checked there at all; placeholders still
    are, because an unresolved one would reach the report.
    """
    if not isinstance(text, str) or not text:
        return []
    import difflib
    problems = []
    for raw in ANY_BRACED.findall(text):
        key = raw.strip()
        if key in evidence:
            continue
        near = difflib.get_close_matches(key, list(evidence), n=1, cutoff=0.6)
        hint = f". Did you mean {{{{{near[0]}}}}}?" if near else ""
        if re.fullmatch(r"-?\d+(?:\.\d+)?", key):
            problems.append(f"{{{{{key}}}}} puts a NUMBER inside the braces. Put the evidence KEY "
                            f"(the text left of '=' in the EVIDENCE list) inside the braces instead")
        else:
            problems.append(f"unknown evidence key {{{{{key}}}}}: only keys from the EVIDENCE list "
                            f"of THIS section may be cited{hint}")
    if prescriptive:
        return problems
    stripped = PLACEHOLDER.sub(" ", text)
    stripped = SCALE.sub(" ", stripped)
    for m in NUMBER.finditer(stripped):
        before = stripped[max(0, m.start() - 16):m.start()]
        after = stripped[m.end():m.end() + 16]
        if REFERENCE_BEFORE.search(before):
            continue
        if prescriptive and DOSE_AFTER.match(after):
            continue
        problems.append(f"number '{m.group(0)}' typed outside an evidence placeholder "
                        f"(...{before.strip()[-12:]} {m.group(0)} {after.strip()[:12]}...)")
    return problems


def walk_strings(obj: Any, path: str = "") -> Iterable[tuple[str, str]]:
    """Yield (json-path, string) for every string in a nested structure."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, f"{path}[{i}]")


def check_object(obj: Any, evidence: dict, prescriptive_fields: set[str] | None = None,
                 skip_fields: set[str] | None = None) -> list[str]:
    """Ground every string in a section's JSON output.

    prescriptive_fields: leaf field names where training doses are allowed
    skip_fields: leaf field names holding enums or ids, not prose
    """
    prescriptive_fields = prescriptive_fields or set()
    skip_fields = skip_fields or set()
    out = []
    for path, text in walk_strings(obj):
        leaf = re.sub(r"\[\d+\]$", "", path.split(".")[-1])
        if leaf in skip_fields:
            continue
        for p in check_text(text, evidence, prescriptive=leaf in prescriptive_fields):
            out.append(f"{path}: {p}")
    # evidence_keys arrays must reference real keys
    for path, text in walk_strings(obj):
        if re.search(r"evidence_keys\[\d+\]$", path) and text not in evidence:
            out.append(f"{path}: unknown evidence key '{text}'")
    return out


def format_value(entry: dict) -> str:
    v, unit = entry.get("value"), entry.get("units") or ""
    if v is None:
        return "n/a"
    if isinstance(v, float):
        txt = f"{v:g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.4f}"
    else:
        txt = str(v)
    short = {"shoulder widths": "SW", "shoulder widths/s": "SW/s"}.get(unit, unit)
    if not short:
        return txt
    return f"{txt}{short}" if short == "%" else f"{txt} {short}"


def substitute(text: str, evidence: dict) -> str:
    if not isinstance(text, str):
        return text
    return PLACEHOLDER.sub(lambda m: format_value(evidence.get(m.group(1), {})), text)


def substitute_all(obj: Any, evidence: dict) -> Any:
    if isinstance(obj, str):
        return substitute(obj, evidence)
    if isinstance(obj, dict):
        return {k: substitute_all(v, evidence) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_all(v, evidence) for v in obj]
    return obj


# ---------------------------------------------------------------- auto-linking
# Local models often restate a value they were shown ("mean of 164.3 deg")
# instead of citing its key, and keep doing so across retries. When a typed
# number matches a value from THIS call's evidence list, and the match is
# unambiguous, it is replaced by the placeholder for that key. The rendered text
# is unchanged; provenance becomes traceable. Everything is logged.
#
# Deliberately NOT linked (they stay violations and go back to the model):
#   * thresholds and approximations: "< 3%", "~164", "about 10"
#   * integers that merely round to a value (only exact integer values match)
#   * anything with two or more equally good candidate keys

KEY_PAT = (r"(?:L|R|both|all|session|quality|benchmark|gait|asym|drift|early|late|risk|screen|events|event\d+)"
           r"\.[A-Za-z0-9_.\-]*[A-Za-z0-9_]")
KEY_EQ = re.compile(r"`?\{?\{?\s*(" + KEY_PAT + r")\s*\}?\}?\s*=\s*-?\d+(?:\.\d+)?\s*"
                    r"(?:deg(?:rees)?|°|SW/s|SW|ms|s|%|fps|landmarks|strides|steps/min|km/h|m)?`?")
BARE_KEY = re.compile(r"(?<![{\w.])(" + KEY_PAT + r")(?![\w}])")
UNIT_AFTER = re.compile(r"^\s*(?:deg(?:rees)?\b|°|SW/s\b|SW\b|shoulder[- ]widths?(?:/s)?|leg[- ]lengths?|%|fps\b|ms\b|s\b|sec(?:onds?)?\b)",
                        re.I)
QUALIFIER_BEFORE = re.compile(r"(<|>|≤|≥|~|≈|\babout|\bapprox\w*|\baround|\bunder|\bover|\bbelow|\babove|"
                              r"\bless than|\bmore than|\bat least|\bat most|\bup to|\bnearly|\broughly)\s*$", re.I)
STAT_HINTS = [(re.compile(r"\b(pct|percent|change)\b", re.I), ".change_pct"),
              (re.compile(r"\b(sd|std|standard deviation)\b", re.I), ".sd"),
              (re.compile(r"\b(cv|coefficient of variation)\b", re.I), ".cv_pct"),
              (re.compile(r"\brange\b", re.I), ".range"),
              (re.compile(r"\b(mean|average|avg)\b", re.I), ".mean"),
              (re.compile(r"\bkey[- ]frame\b", re.I), ".at_key_frame"),
              (re.compile(r"\b(minimum|min)\b", re.I), ".min"),
              (re.compile(r"\b(maximum|max|peak)\b", re.I), ".max"),
              (re.compile(r"\bduration\b", re.I), ".duration_s")]


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def _measure_words(key: str) -> set[str]:
    parts = key.split(".")
    words = set()
    for p in parts[1:-1] if len(parts) > 2 else parts:
        words |= {w for w in re.split(r"[_\-]", p.lower()) if len(w) > 2 and w not in ("deg", "norm", "pct")}
    return words


def _candidates(num: str, before: str, subset: dict) -> list[str]:
    typed = float(num)
    d = _decimals(num)
    out = []
    for k, e in subset.items():
        v = e.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if d == 0:
            if float(v) == typed and float(v).is_integer():
                out.append(k)
        elif round(float(v), d) == typed:
            out.append(k)
    if len(out) <= 1:
        return out
    ctx = before[-40:]
    for rx, suffix in STAT_HINTS:
        if rx.search(ctx):
            narrowed = [k for k in out if k.endswith(suffix) or f"{suffix}." in k]
            if narrowed:
                out = narrowed
                break
    if len(out) <= 1:
        return out
    window = before[-90:].lower()
    scored = sorted(((sum(w in window for w in _measure_words(k)), k) for k in out), reverse=True)
    if scored[0][0] > 0 and scored[0][0] > scored[1][0]:
        return [scored[0][1]]
    return out


def link_text(text: str, subset: dict, evidence: dict) -> tuple[str, list[str]]:
    if not isinstance(text, str) or not text:
        return text, []
    log = []

    def eq_sub(m):
        k = m.group(1)
        if k in evidence:
            log.append(f"'{m.group(0).strip()}' -> {{{{{k}}}}}")
            return "{{" + k + "}}"
        return m.group(0)
    text = KEY_EQ.sub(eq_sub, text)

    def bare_sub(m):
        k = m.group(1)
        if k in evidence:
            log.append(f"bare key {k} -> {{{{{k}}}}}")
            return "{{" + k + "}}"
        return m.group(0)
    # only outside existing placeholders
    parts = re.split(r"(\{\{[^}]*\}\})", text)
    parts = [p if p.startswith("{{") else BARE_KEY.sub(bare_sub, p) for p in parts]
    text = "".join(parts)

    out, pos = [], 0
    stripped_scale = SCALE
    for seg in re.split(r"(\{\{[^}]*\}\})", text):
        if seg.startswith("{{"):
            out.append(seg)
            continue
        result, last = [], 0
        for m in NUMBER.finditer(seg):
            before, after = seg[:m.start()], seg[m.end():]
            if REFERENCE_BEFORE.search(before[-16:]) or QUALIFIER_BEFORE.search(before[-14:]) \
                    or stripped_scale.match(seg[m.start():m.start() + 6]):
                continue
            cands = _candidates(m.group(0), before, subset)
            if len(cands) != 1:
                continue
            unit = UNIT_AFTER.match(after)
            end = m.end() + (unit.end() if unit else 0)
            result.append(seg[last:m.start()])
            result.append("{{" + cands[0] + "}}")
            log.append(f"'{seg[m.start():end].strip()}' -> {{{{{cands[0]}}}}}")
            last = end
        result.append(seg[last:])
        out.append("".join(result))
    return "".join(out), log


def link_object(obj, subset: dict, evidence: dict, skip_fields: set, prescriptive_fields: set,
                path: str = "") -> tuple[object, list[str]]:
    """Apply link_text to every prose field. Returns (new object, log)."""
    log: list[str] = []
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if k in skip_fields:
                new[k] = v
                continue
            if isinstance(v, str) and k not in prescriptive_fields:
                nv, lg = link_text(v, subset, evidence)
                log += [f"{path}.{k}: {x}" if path else f"{k}: {x}" for x in lg]
                new[k] = nv
            else:
                nv, lg = link_object(v, subset, evidence, skip_fields, prescriptive_fields,
                                     f"{path}.{k}" if path else k)
                log += lg
                new[k] = nv
        return new, log
    if isinstance(obj, list):
        new_list = []
        for i, v in enumerate(obj):
            if isinstance(v, str):
                nv, lg = link_text(v, subset, evidence)
                log += [f"{path}[{i}]: {x}" for x in lg]
                new_list.append(nv)
            else:
                nv, lg = link_object(v, subset, evidence, skip_fields, prescriptive_fields, f"{path}[{i}]")
                log += lg
                new_list.append(nv)
        return new_list, log
    return obj, log
