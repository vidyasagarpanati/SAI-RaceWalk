"""Per-run state manifest: status, hashes, checks, resume.

runs/<run_id>/state.json is the human-readable source of truth for what has been
done. A step is skipped on resume only when its status is DONE and its recorded
input hash still matches. Nothing is ever deleted; a rerun writes a new record.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from racewalk.contracts import StepResult
from racewalk.io_guard import guarded_open

PENDING, RUNNING, DONE, FAILED, SKIPPED = "PENDING", "RUNNING", "DONE", "FAILED", "SKIPPED"

STEP_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
STEP_NAMES = {
    "S0": "ingest", "S1": "frames", "S2": "pose", "S3": "kinematics",
    "S4": "gait", "S5": "stats", "S6": "annotate", "S7": "video",
    "S8": "narrate", "S9": "verify", "S10": "render",
}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunState:
    run_dir: Path
    data: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.run_dir / "state.json"

    @classmethod
    def load_or_create(cls, run_dir: Path, **meta) -> "RunState":
        run_dir = Path(run_dir)
        state_path = run_dir / "state.json"
        if state_path.is_file():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            data.setdefault("steps", {})
            data.update({k: v for k, v in meta.items() if v is not None})
        else:
            data = {"created": _now(), "steps": {}, **meta}
        obj = cls(run_dir=run_dir, data=data)
        obj.save()
        return obj

    def save(self) -> None:
        with guarded_open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)

    # -- step records ----------------------------------------------------
    def record(self, step: str) -> dict:
        return self.data["steps"].setdefault(step, {"status": PENDING})

    def status(self, step: str) -> str:
        return self.record(step).get("status", PENDING)

    def should_skip(self, step: str, input_hash: str | None) -> bool:
        rec = self.record(step)
        if rec.get("status") != DONE:
            return False
        if input_hash is None:
            return True
        return rec.get("input_hash") == input_hash

    def start(self, step: str, input_hash: str | None = None) -> None:
        rec = self.record(step)
        rec.update({
            "name": STEP_NAMES.get(step, step),
            "status": RUNNING,
            "started": _now(),
            "input_hash": input_hash,
            "finished": None,
            "error": None,
        })
        self.save()

    def finish(self, step: str, result: StepResult) -> None:
        rec = self.record(step)
        rec.update({
            "status": DONE if result.passed else FAILED,
            "finished": _now(),
            "outputs": result.outputs,
            "stats": result.stats,
            "checks": [
                {"name": c.name, "ok": c.ok, "severity": c.severity, "detail": c.detail}
                for c in result.checks
            ],
            "n_failures": len(result.failures),
            "n_warnings": len(result.warnings),
        })
        self.save()

    def fail(self, step: str, error: str) -> None:
        rec = self.record(step)
        rec.update({"status": FAILED, "finished": _now(), "error": error})
        self.save()

    def upstream_ok(self, step: str) -> tuple[bool, str]:
        """Every earlier step must be DONE before ``step`` may run."""
        idx = STEP_ORDER.index(step)
        for prior in STEP_ORDER[:idx]:
            st = self.status(prior)
            if st != DONE:
                return False, f"{prior} ({STEP_NAMES[prior]}) is {st}, expected DONE"
        return True, ""

    def render_table(self) -> str:
        total = sum(rec.get("duration_s") or 0 for rec in self.data["steps"].values())
        lines = [f"run_id: {self.data.get('run_id')}",
                 f"video : {self.data.get('video_path')}",
                 f"config: {self.data.get('config_hash')}",
                 "",
                 f"{'step':<5} {'name':<12} {'status':<8} {'checks':<8} {'took':>10}  finished"]
        for s in STEP_ORDER:
            rec = self.data["steps"].get(s, {})
            checks = rec.get("checks", [])
            ok = sum(1 for c in checks if c.get("ok"))
            cs = f"{ok}/{len(checks)}" if checks else "-"
            d = rec.get("duration_s")
            took = f"{d / 60:.1f} min" if d and d >= 60 else (f"{d:.0f}s" if d else "-")
            lines.append(f"{s:<5} {STEP_NAMES[s]:<12} {rec.get('status', PENDING):<8} "
                         f"{cs:<8} {took:>10}  {rec.get('finished') or ''}")
        lines.append("")
        lines.append(f"total measured time: {total / 60:.1f} min")
        return "\n".join(lines)
