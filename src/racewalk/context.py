"""The object every step receives. Carries paths, config, session metadata, state."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from racewalk.config import Config
from racewalk.runstate import RunState

FRONTAL = {"rear", "front"}
SAGITTAL = {"side_left", "side_right"}


@dataclass
class Context:
    cfg: Config
    run_id: str
    run_dir: Path
    video_path: Path
    session: dict[str, Any] = field(default_factory=dict)
    state: RunState | None = None
    llm: Any = None          # injected LLM client (tests use a fake); S8 builds Ollama otherwise

    # -- conventional artefact locations --------------------------------
    @property
    def frames_dir(self) -> Path:
        return self.run_dir / "frames"

    @property
    def key_frames_dir(self) -> Path:
        return self.run_dir / "06_frames"

    @property
    def video_dir(self) -> Path:
        return self.run_dir / "07_video"

    @property
    def narrative_dir(self) -> Path:
        return self.run_dir / "08_narrative"

    def artefact(self, name: str) -> Path:
        return self.run_dir / name

    def read_json(self, name: str) -> dict:
        p = self.artefact(name)
        if not p.is_file():
            raise FileNotFoundError(
                f"Expected artefact {name} is missing from {self.run_dir}. "
                f"Run the earlier step first."
            )
        return json.loads(p.read_text(encoding="utf-8"))

    def write_json(self, name: str, payload: Any) -> Path:
        from racewalk.io_guard import guarded_open
        p = self.artefact(name)
        with guarded_open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return p

    # -- camera view ------------------------------------------------------
    @property
    def camera_view(self) -> str:
        return (self.session.get("camera_view") or "rear").lower()

    @property
    def view_class(self) -> str:
        """frontal (rear or front), sagittal (either side) or oblique."""
        v = self.camera_view
        return "frontal" if v in FRONTAL else "sagittal" if v in SAGITTAL else "oblique"
