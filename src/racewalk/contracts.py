"""Step contracts: every agent declares INPUT, OUTPUT and VERIFY.

A step never returns 'done'. It returns a StepResult carrying an explicit list of
checks. The next step refuses to start if any upstream check failed. This is the
'verification steps passed on to the agents' requirement, made mechanical.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

FAIL = "FAIL"
WARN = "WARN"
INFO = "INFO"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    severity: str = FAIL          # FAIL blocks the pipeline, WARN is recorded only
    evidence: Any = None          # the value the check looked at, for auditability


@dataclass
class StepResult:
    step: str
    outputs: dict[str, str] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str = "", severity: str = FAIL,
              evidence: Any = None) -> "StepResult":
        self.checks.append(Check(name, bool(ok), detail, severity, evidence))
        return self

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == WARN]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "passed": self.passed,
            "outputs": self.outputs,
            "stats": self.stats,
            "checks": [asdict(c) for c in self.checks],
        }

    def summary(self) -> str:
        n_ok = sum(1 for c in self.checks if c.ok)
        return (f"{self.step}: {'PASS' if self.passed else 'FAIL'} "
                f"({n_ok}/{len(self.checks)} checks ok, "
                f"{len(self.warnings)} warnings)")


class StepFailed(RuntimeError):
    def __init__(self, result: StepResult):
        self.result = result
        lines = [f"{result.step} failed its own verification:"]
        for c in result.failures:
            lines.append(f"  [FAIL] {c.name}: {c.detail}")
        super().__init__("\n".join(lines))


class UpstreamFailed(RuntimeError):
    pass


def dump_result(result: StepResult, path) -> None:
    from racewalk.io_guard import guarded_open
    with guarded_open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
