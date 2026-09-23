"""Collect everything needed to identify why MediaPipe's native bindings fail.

    D:\\SAI\\.venv\\Scripts\\python.exe scripts\\diagnose_mediapipe.py

Prints a report. Paste the whole thing. Reads only, changes nothing.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import platform
import sys
import traceback
from pathlib import Path

LINE = "-" * 72


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


PROBE_A = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_framework_bindings', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('PROBE_A OK: _framework_bindings loads with no cv2 in the process')
"""

PROBE_B = """
import importlib.util, sys
import cv2
print('cv2', cv2.__version__, 'loaded first')
spec = importlib.util.spec_from_file_location('_framework_bindings', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('PROBE_B OK: _framework_bindings loads after cv2')
"""

PROBE_C = """
from mediapipe.tasks.python import vision
print('PROBE_C OK: full mediapipe.tasks.python.vision import')
"""


def _probe(name: str, code: str, pyd: str) -> tuple[bool, str]:
    import subprocess
    proc = subprocess.run([sys.executable, "-c", code, pyd],
                          capture_output=True, text=True, timeout=180)
    output = (proc.stdout + proc.stderr).strip()
    ok = proc.returncode == 0
    print(f"\n  [{name}] {'PASS' if ok else 'FAIL'}")
    for line in output.splitlines()[-14:]:
        print("     ", line)
    return ok, output


def isolation(pyd: Path) -> None:
    section("ISOLATION PROBES (each in a fresh interpreter)")
    print("  A: load _framework_bindings with NO cv2 in the process")
    print("  B: import cv2 first, then load _framework_bindings")
    print("  C: the normal import the pipeline performs")
    a, _ = _probe("A", PROBE_A, str(pyd))
    b, _ = _probe("B", PROBE_B, str(pyd))
    c, _ = _probe("C", PROBE_C, str(pyd))

    section("VERDICT")
    try:
        from importlib.metadata import version
        cv_ver = version("opencv-contrib-python")
    except Exception:
        cv_ver = "unknown"
    if c:
        print("  MediaPipe imports correctly in this interpreter:", sys.executable)
        print("  If the pipeline still fails, in order of likelihood:")
        print("    1. The failing run predates the last install. Just run it again.")
        print("    2. `racewalk` on PATH resolves to a different interpreter. Check with:")
        print("         (Get-Command racewalk).Source")
        print("       and compare against the executable above.")
    elif a and not b:
        print(f"  cv2 is the trigger. opencv-contrib-python {cv_ver} loads native DLLs that")
        print("  conflict with MediaPipe's when both are in one process. MediaPipe imports")
        print("  cv2 itself (mediapipe/__init__ -> solutions -> drawing_utils), so the import")
        print("  order cannot be worked around in our code. The fix is the OpenCV version.")
        print("")
        print("  Try, in this order, stopping at the first that works:")
        print("    .\\.venv\\Scripts\\python.exe -m pip install opencv-contrib-python==4.10.0.84")
        print("    .\\.venv\\Scripts\\python.exe -m pip install opencv-contrib-python==4.9.0.80")
        print("    .\\.venv\\Scripts\\python.exe -m pip install opencv-contrib-python==4.8.1.78")
        print("  Re-run `racewalk diagnose` after each.")
    elif not a:
        print("  _framework_bindings cannot load even on its own, with no cv2 present.")
        print("  This is a MediaPipe wheel or runtime problem, not an OpenCV clash.")
        print("  Next step: try a different mediapipe build.")
        print("    .\\.venv\\Scripts\\python.exe -m pip install --no-deps --force-reinstall mediapipe==0.10.18")
    elif a and b and not c:
        print("  The native bindings are fine. Something else in mediapipe's package")
        print("  import chain is failing. The PROBE_C traceback above names it.")
    print("")


def main() -> int:
    section("INTERPRETER")
    print("executable :", sys.executable)
    print("version    :", sys.version.replace("\n", " "))
    print("arch       :", platform.machine(), platform.architecture()[0])
    print("platform   :", platform.platform())
    if sys.platform == "win32":
        print("win32_ver  :", platform.win32_ver())
        try:
            print("edition    :", platform.win32_edition())
            print("is_iot     :", platform.win32_is_iot())
        except Exception as exc:
            print("edition    : unavailable", exc)

    section("PACKAGE VERSIONS")
    from importlib.metadata import PackageNotFoundError, distributions, version
    watch = ["mediapipe", "numpy", "protobuf", "flatbuffers", "attrs",
             "opencv-contrib-python", "opencv-python", "opencv-python-headless",
             "opencv-contrib-python-headless", "sentencepiece", "sounddevice",
             "jax", "jaxlib", "absl-py", "matplotlib"]
    for name in watch:
        try:
            print(f"{name:<32} {version(name)}")
        except PackageNotFoundError:
            print(f"{name:<32} not installed")
    installed_opencv = sorted({
        d.metadata["Name"] for d in distributions()
        if (d.metadata["Name"] or "").lower().startswith("opencv")})
    print("\nopencv distributions present:", installed_opencv)

    section("VISUAL C++ RUNTIME")
    if sys.platform == "win32":
        for dll in ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
                    "msvcp140_1.dll", "concrt140.dll", "ucrtbase.dll"]:
            try:
                handle = ctypes.CDLL(dll)
                print(f"  ok      {dll}  -> {getattr(handle, '_name', dll)}")
            except OSError as exc:
                print(f"  MISSING {dll}  -> {exc}")
    else:
        print("  not Windows, skipped")

    section("MEDIA FOUNDATION (OpenCV video backends need these)")
    if sys.platform == "win32":
        for dll in ["mf.dll", "mfplat.dll", "mfreadwrite.dll", "mfcore.dll"]:
            try:
                ctypes.CDLL(dll)
                print(f"  ok      {dll}")
            except OSError as exc:
                print(f"  MISSING {dll}  -> {exc}")
        print("  Note: Windows Server and N editions ship without these unless the")
        print("        Media Foundation / Media Feature Pack feature is installed.")
    else:
        print("  not Windows, skipped")

    section("STEP-BY-STEP IMPORT")
    for mod in ["numpy", "cv2", "google.protobuf", "flatbuffers",
                "mediapipe", "mediapipe.python",
                "mediapipe.python._framework_bindings",
                "mediapipe.tasks", "mediapipe.tasks.python",
                "mediapipe.tasks.python.vision"]:
        try:
            m = importlib.import_module(mod)
            where = getattr(m, "__file__", "(namespace)")
            print(f"  ok      {mod}\n            {where}")
        except Exception as exc:
            print(f"  FAILED  {mod}")
            print(f"            {type(exc).__name__}: {exc}")
            break

    section("NATIVE EXTENSION FILE")
    try:
        import mediapipe
        root = Path(mediapipe.__file__).parent
    except Exception:
        for p in sys.path:
            cand = Path(p) / "mediapipe"
            if cand.is_dir():
                root = cand
                break
        else:
            print("  mediapipe package directory not found on sys.path")
            return 1
    print("  package dir:", root)
    pyds = sorted(root.rglob("_framework_bindings*.pyd")) + \
           sorted(root.rglob("_framework_bindings*.so"))
    if not pyds:
        print("  NO _framework_bindings binary found. The wheel is incomplete.")
    for pyd in pyds:
        print(f"  {pyd}  ({pyd.stat().st_size / 1e6:.1f} MB)")
        if sys.platform == "win32":
            try:
                ctypes.WinDLL(str(pyd))
                print("    ctypes.WinDLL loaded it cleanly "
                      "(so the failure is in Python-level init, not the DLL itself)")
            except OSError as exc:
                print(f"    ctypes.WinDLL failed: winerror={getattr(exc, 'winerror', None)} {exc}")

    section("DLL SEARCH PATH")
    print("  PATH entries containing 'python', 'venv', 'ffmpeg' or 'cuda':")
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        low = entry.lower()
        if any(k in low for k in ("python", "venv", "ffmpeg", "cuda", "nvidia")):
            print("   ", entry)

    section("OPENCV DETAIL")
    try:
        import cv2
        print("  cv2 version :", cv2.__version__)
        print("  cv2 file    :", cv2.__file__)
        cv2_dir = Path(cv2.__file__).parent
        dlls = sorted(cv2_dir.rglob("*.dll"))
        print(f"  {len(dlls)} DLLs under {cv2_dir}")
        for d in dlls[:12]:
            print(f"    {d.name}  ({d.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print("  cv2 not importable:", type(exc).__name__, exc)

    section("FULL TRACEBACK OF THE FAILING IMPORT")
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        print("  It imported successfully in this process. "
              "If the pipeline still fails, the venv it runs in differs from this one.")
    except Exception:
        traceback.print_exc()

    if pyds:
        isolation(pyds[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
