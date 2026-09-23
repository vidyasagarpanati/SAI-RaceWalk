"""Write whitelist and read-only source access.

Guarantees enforced here:
  1. The source video is opened read-only and its directory is never writable.
  2. Every write in the pipeline routes through ``guarded_open`` / ``guarded_path``.
  3. There is no delete primitive anywhere in this package. Nothing the pipeline
     touches is ever removed. Cleanup is a human decision.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Iterable

_WRITABLE_ROOTS: list[Path] = []


class WriteGuardError(PermissionError):
    """Raised when something tries to write outside the whitelist."""


def configure(writable_roots: Iterable[Path]) -> None:
    """Declare the only directories this process may write into."""
    global _WRITABLE_ROOTS
    roots = []
    for r in writable_roots:
        p = Path(r).resolve()
        p.mkdir(parents=True, exist_ok=True)
        roots.append(p)
    _WRITABLE_ROOTS = roots


def writable_roots() -> list[Path]:
    return list(_WRITABLE_ROOTS)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def guarded_path(path: str | os.PathLike) -> Path:
    """Resolve ``path`` and assert it is inside a whitelisted writable root."""
    if not _WRITABLE_ROOTS:
        raise WriteGuardError(
            "io_guard.configure() was never called. Refusing to write anything."
        )
    p = Path(path)
    # Resolve the parent, because the file itself may not exist yet.
    resolved = (p.parent.resolve() / p.name) if p.parent.exists() else Path(os.path.abspath(p))
    if not any(_is_inside(resolved, root) for root in _WRITABLE_ROOTS):
        raise WriteGuardError(
            f"Refusing to write outside the whitelist.\n"
            f"  target : {resolved}\n"
            f"  allowed: {[str(r) for r in _WRITABLE_ROOTS]}"
        )
    return resolved


def guarded_open(path: str | os.PathLike, mode: str = "w", **kwargs) -> IO:
    """``open`` that refuses any write mode outside the whitelist."""
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        target = guarded_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return open(target, mode, **kwargs)
    return open(path, mode, **kwargs)


def open_source_readonly(path: str | os.PathLike) -> IO[bytes]:
    """Open a source asset strictly read-only.

    Uses os.open with O_RDONLY so the intent is enforced by the kernel, not by
    convention. Windows honours this too.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Source file not found: {p}")
    fd = os.open(p, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    return os.fdopen(fd, "rb")


def assert_source_not_writable(source: str | os.PathLike) -> None:
    """Fail loudly if the source video's directory is on the writable whitelist."""
    src_dir = Path(source).resolve().parent
    for root in _WRITABLE_ROOTS:
        if _is_inside(src_dir, root) or _is_inside(root, src_dir):
            raise WriteGuardError(
                "The source video directory overlaps a writable root. "
                "Move runs/ and outputs/ somewhere else, or move the video.\n"
                f"  source dir : {src_dir}\n"
                f"  writable   : {root}"
            )
