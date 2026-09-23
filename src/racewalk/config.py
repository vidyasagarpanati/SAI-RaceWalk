"""Configuration loading and the reproducibility hash."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash a file without ever opening it for writing."""
    from racewalk.io_guard import open_source_readonly
    h = hashlib.sha256()
    with open_source_readonly(path) as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass
class Paths:
    root: Path
    config_dir: Path
    schemas_dir: Path
    templates_dir: Path
    runs_dir: Path
    outputs_dir: Path
    models_dir: Path

    @classmethod
    def from_root(cls, root: Path, cfg: dict) -> "Paths":
        p = cfg.get("paths", {})
        return cls(
            root=root,
            config_dir=root / "config",
            schemas_dir=root / "schemas",
            templates_dir=root / "templates",
            runs_dir=root / p.get("runs_dir", "runs"),
            outputs_dir=root / p.get("outputs_dir", "outputs"),
            models_dir=root / p.get("models_dir", "models"),
        )


@dataclass
class Config:
    root: Path
    pipeline: dict
    phase_rules: dict
    benchmarks: dict
    risk_rules: dict
    guidance: dict
    paths: Paths
    prompts: dict
    config_hash: str

    def slice_hash(self, keys: list[str]) -> str:
        """Hash only the configuration a step reads.

        Keys are dotted paths into pipeline.yaml, or the whole-file names
        "phase_rules" and "benchmarks". Changing a phase threshold must not
        invalidate pose estimation, so each step hashes its own slice.
        """
        picked = {}
        for key in sorted(keys):
            if key == "phase_rules":
                picked[key] = self.phase_rules
            elif key == "risk_rules":
                picked[key] = self.risk_rules
            elif key == "guidance":
                picked[key] = self.guidance
            elif key == "benchmarks":
                picked[key] = self.benchmarks
            elif key == "prompts":
                picked[key] = self.prompts
            else:
                picked[key] = self.get(key)
        return sha256_text(_canonical(picked))[:16]

    def get(self, dotted: str, default=None):
        node: Any = self.pipeline
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until a directory containing config/pipeline.yaml."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "pipeline.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root (a directory containing config/pipeline.yaml). "
        "Run the CLI from inside the repository, or pass --root."
    )


def load_config(root: Path | None = None) -> Config:
    root = Path(root).resolve() if root else find_project_root()
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
    phase_rules = yaml.safe_load((root / "config" / "phase_rules.yaml").read_text(encoding="utf-8"))
    benchmarks = json.loads((root / "config" / "benchmarks.json").read_text(encoding="utf-8"))
    risk_rules = yaml.safe_load((root / "config" / "risk_rules.yaml").read_text(encoding="utf-8"))
    guidance = yaml.safe_load((root / "config" / "guidance.yaml").read_text(encoding="utf-8"))
    prompts = {}
    pdir = root / "config" / "prompts"
    if pdir.is_dir():
        for f in sorted(pdir.iterdir()):
            if f.suffix == ".md":
                prompts[f.stem] = f.read_text(encoding="utf-8")
            elif f.suffix in (".yaml", ".yml"):
                prompts[f.stem] = yaml.safe_load(f.read_text(encoding="utf-8"))

    # The reproducibility hash covers everything that can change a number.
    config_hash = sha256_text(_canonical({
        "pipeline": pipeline,
        "phase_rules": phase_rules,
        "benchmarks": benchmarks,
        "risk_rules": risk_rules,
        "guidance": guidance,
        "prompts": prompts,
    }))[:16]

    return Config(
        root=root,
        pipeline=pipeline,
        phase_rules=phase_rules,
        benchmarks=benchmarks,
        risk_rules=risk_rules,
        guidance=guidance,
        paths=Paths.from_root(root, pipeline),
        prompts=prompts,
        config_hash=config_hash,
    )
