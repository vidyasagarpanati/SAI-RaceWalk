"""Versioned output names. Nothing in outputs/ is ever overwritten."""
from __future__ import annotations

import re
from pathlib import Path


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "Athlete"


def next_versioned(directory: Path, stem: str, suffix: str) -> Path:
    """<stem>_v01<suffix>, then _v02, ... Returns the first name not yet taken."""
    directory.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(stem)}_v(\d+){re.escape(suffix)}$")
    taken = [int(m.group(1)) for p in directory.iterdir() if (m := pat.match(p.name))]
    return directory / f"{stem}_v{(max(taken) + 1 if taken else 1):02d}{suffix}"
