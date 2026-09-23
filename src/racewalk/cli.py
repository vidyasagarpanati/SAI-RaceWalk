"""Command line interface.

    racewalk doctor
    racewalk init-session --video "D:\\walks\\athlete_01.mp4"
    racewalk run --video "D:\\walks\\athlete_01.mp4"
    racewalk run --video ... --to S5            # stop after the evidence file
    racewalk run --video ... --only S6          # rerun one step
    racewalk run --video ... --from S8 --force  # redo the narrative
    racewalk status
    racewalk status <run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from racewalk import io_guard
from racewalk.config import load_config, sha256_file
from racewalk.context import Context
from racewalk.contracts import StepFailed, UpstreamFailed
from racewalk.runstate import STEP_NAMES, STEP_ORDER, RunState


def _bootstrap(root: Path | None):
    cfg = load_config(root)
    io_guard.configure([
        cfg.paths.runs_dir,
        cfg.paths.outputs_dir,
        cfg.root / "sessions",
        cfg.paths.models_dir,
    ])
    return cfg


def _session_path(cfg, video: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return cfg.root / "sessions" / f"{video.stem}.json"


def _load_session(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"No session metadata at {path}.\n"
            f"Create one with:  racewalk init-session --video <path to video>\n"
            f"then fill in athlete_name, camera_view and, if known, treadmill_speed_kmh."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _make_run_id(cfg, video: Path, digest: str) -> str:
    """One run directory per video. Config changes are handled per step by the
    cache keys in runner.STEP_DEPS, so tuning a threshold re-runs only the steps
    that read it, inside the same run."""
    return f"{video.stem}__{digest[:8]}"


def cmd_init_session(args) -> int:
    cfg = _bootstrap(args.root)
    video = Path(args.video)
    target = _session_path(cfg, video, args.session)
    if target.exists() and not args.force:
        print(f"Already exists: {target}\nPass --force to overwrite.")
        return 1
    template = json.loads((cfg.root / "session.example.json").read_text(encoding="utf-8"))
    template["athlete_name"] = args.athlete or video.stem
    with io_guard.guarded_open(target, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=2)
    print(f"Wrote {target}\nEdit it, then run:  racewalk run --video \"{video}\"")
    return 0


def cmd_doctor(args) -> int:
    import shutil
    cfg = _bootstrap(args.root)
    rows: list[tuple[str, bool, str]] = []

    rows.append(("python", sys.version_info[:2] >= (3, 11) and sys.version_info[:2] < (3, 13),
                 f"{sys.version.split()[0]} (need 3.11 or 3.12 for mediapipe wheels)"))

    for mod in ["cv2", "numpy", "pandas", "pyarrow", "yaml",
                "jsonschema", "jinja2", "httpx", "langgraph"]:
        try:
            __import__(mod)
            rows.append((mod, True, "installed"))
        except Exception as exc:  # noqa: BLE001
            rows.append((mod, False, f"missing: {exc}"))

    # MediaPipe runs in an isolated worker process (see racewalk/pose_worker.py),
    # so test it exactly that way: a fresh interpreter, nothing else loaded.
    import subprocess
    probe = subprocess.run([sys.executable, "-m", "racewalk.pose_worker", "--selftest"],
                           capture_output=True, text=True, timeout=180)
    if probe.returncode == 0:
        rows.append(("mediapipe worker", True, probe.stdout.strip().splitlines()[-1]))
    else:
        tail = (probe.stderr or probe.stdout).strip().splitlines()[-1:] or ["no output"]
        rows.append(("mediapipe worker", False, tail[0]))

    # Informational: does pyarrow-then-mediapipe fail on this machine? If so, the
    # isolation above is what keeps the pipeline working.
    clash = subprocess.run(
        [sys.executable, "-c", "import pyarrow; from mediapipe.tasks.python import vision"],
        capture_output=True, text=True, timeout=180)
    rows.append(("pyarrow+mediapipe", True,
                 "can share a process" if clash.returncode == 0 else
                 "CONFLICT when loaded in one process (expected on some Windows builds; "
                 "handled by running pose in an isolated worker)"))

    # Only one distribution may own site-packages/cv2.
    try:
        from importlib.metadata import distributions
        opencv_pkgs = sorted({
            d.metadata["Name"] for d in distributions()
            if (d.metadata["Name"] or "").lower().startswith("opencv")
        })
        rows.append(("opencv packages", len(opencv_pkgs) == 1,
                     f"{opencv_pkgs} (exactly one must be installed; mediapipe needs "
                     f"opencv-contrib-python)" if opencv_pkgs != ["opencv-contrib-python"]
                     else "opencv-contrib-python only, correct"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("opencv packages", False, f"could not enumerate: {exc}"))

    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.CDLL("vcruntime140_1.dll")
            rows.append(("vcruntime", True, "Visual C++ 2015-2022 x64 runtime present"))
        except OSError:
            rows.append(("vcruntime", False,
                         "vcruntime140_1.dll not loadable. Install the Microsoft Visual "
                         "C++ 2015-2022 x64 redistributable: "
                         "winget install Microsoft.VCRedist.2015+.x64"))

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    rows.append(("ffmpeg", bool(ffmpeg), ffmpeg or "not found"))
    rows.append(("ffprobe", bool(shutil.which("ffprobe")),
                 shutil.which("ffprobe") or "not found (OpenCV fallback will be used)"))

    model = cfg.paths.models_dir / cfg.get("paths.pose_model_file")
    rows.append(("pose model", model.is_file(),
                 str(model) if model.is_file() else f"missing: {model}"))

    base = cfg.get("llm.base_url")
    try:
        import httpx
        r = httpx.get(f"{base}/api/tags", timeout=5)
        tags = [m["name"] for m in r.json().get("models", [])]
        rows.append(("ollama", True, f"{base} -> {len(tags)} models: {', '.join(tags[:6])}"))
        configured = cfg.get("llm.model")
        rows.append(("llm.model", bool(configured) and configured in tags,
                     f"configured={configured!r}. Set config/pipeline.yaml llm.model to one of the tags above."))
    except Exception as exc:  # noqa: BLE001
        rows.append(("ollama", False, f"{base} unreachable: {exc}"))

    if getattr(args, "llm", False):
        # One tiny structured-output call through the same client S8 uses.
        from racewalk.llm import OllamaClient
        try:
            # Cache inside runs/ so the write guard allows it (a system temp dir
            # is outside the whitelist and is correctly refused).
            probe_dir = io_guard.guarded_path(cfg.paths.runs_dir / "_doctor_llm")
            probe_dir.mkdir(parents=True, exist_ok=True)
            client = OllamaClient(cfg, probe_dir)
            caps = client.capabilities()
            rows.append(("model capabilities", True, ", ".join(sorted(caps)) or "none reported"))
            rows.append(("model vision", "vision" in caps,
                         "key frames will be sent" if "vision" in caps else
                         "no vision: image-dependent items will be NOT RELIABLY ASSESSABLE"))
            import time as _t
            from racewalk.grounding import check_text, keys_in
            fake_ev = {"L.LOADING.knee_deg.min": {"value": 171.5, "units": "deg"},
                       "both.stride.trunk_incl_range_deg.mean": {"value": 2.1, "units": "deg"}}
            user = ("SECTION: doctor_probe\n\nINSTRUCTIONS\nWrite one sentence describing the left "
                    "knee and the trunk, citing both values.\n\n"
                    "EVIDENCE (cite values ONLY as {{key}})\n"
                    "{{L.LOADING.knee_deg.min}} = 171.5 deg [HIGH]\n"
                    "{{both.stride.trunk_incl_range_deg.mean}} = 2.1 deg [HIGH]")
            t0 = _t.time()
            out = client.chat_json(
                cfg.prompts.get("system", "Reply with JSON only."), user,
                {"type": "object", "properties": {"sentence": {"type": "string"},
                 "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
                 "required": ["sentence", "confidence"], "additionalProperties": False})
            sentence = out.get("sentence", "")
            problems = check_text(sentence, fake_ev)
            ok = not problems and len(keys_in(sentence)) >= 1
            rows.append(("structured output", True,
                         f"valid JSON in {_t.time() - t0:.0f}s "
                         f"({client.usage['prompt_tokens']}+{client.usage['completion_tokens']} tokens)"))
            rows.append(("evidence citation", ok,
                         f"reply: {sentence!r}" + ("" if ok else f"  PROBLEMS: {problems or ['no placeholder used']}")))
        except Exception as exc:  # noqa: BLE001
            rows.append(("structured output", False, f"{type(exc).__name__}: {exc}"))

    width = max(len(name) for name, _, _ in rows)
    for name, ok, detail in rows:
        print(f"[{'ok ' if ok else 'XX '}] {name:<{width}}  {detail}")
    blocking = [n for n, ok, _ in rows if not ok and n not in ("ffprobe", "ollama", "llm.model",
                                                                   "model vision", "pyarrow+mediapipe")]
    if blocking:
        print(f"\nBlocking problems: {blocking}")
        print("Run scripts\\bootstrap.ps1 to fix the environment.")
        return 1
    print("\nEnvironment is ready for the deterministic steps (S0 to S7).")
    return 0


def cmd_run(args) -> int:
    from racewalk.graph import describe, run_graph
    from racewalk.runner import execute_step, plan_steps

    cfg = _bootstrap(args.root)
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    session = _load_session(_session_path(cfg, video, args.session))
    digest = sha256_file(video)
    run_id = args.run_id or _make_run_id(cfg, video, digest)
    run_dir = cfg.paths.runs_dir / run_id
    io_guard.guarded_path(run_dir).mkdir(parents=True, exist_ok=True)

    state = RunState.load_or_create(
        run_dir, run_id=run_id, video_path=str(video),
        video_sha256=digest, config_hash=cfg.config_hash,
    )
    ctx = Context(cfg=cfg, run_id=run_id, run_dir=run_dir, video_path=video,
                  session=session, state=state)

    steps = plan_steps(args.from_step, args.to_step, args.only)
    print(f"run_id : {run_id}")
    print(f"video  : {video.name}  sha256 {digest[:16]}")
    print(f"config : {cfg.config_hash}")
    print(f"steps  : {describe(steps)}\n")

    import time as _t
    run_started = _t.time()

    def _summary():
        print("\n" + state.render_table())
        print(f"wall clock: {(_t.time() - run_started) / 60:.1f} min")

    if args.no_graph or args.only:
        for step_id in steps:
            try:
                result = execute_step(ctx, step_id, force=args.force)
            except (StepFailed, UpstreamFailed) as exc:
                print(f"\n{exc}")
                _summary()
                return 2
            except NotImplementedError as exc:
                print(f"\n{exc}")
                return 3
            print(result.summary())
            for c in result.warnings:
                print(f"    [warn] {c.name}: {c.detail}")
        _summary()
        print("\nDone.")
        return 0

    final = run_graph(ctx, steps, force=args.force)
    for step_id in final.get("completed", []):
        print(f"{step_id}: PASS")
    if final.get("failed"):
        print(f"\n{final['failed']} FAILED\n{final.get('error')}")
        _summary()
        return 2
    _summary()
    print("\nDone.")
    return 0


def cmd_status(args) -> int:
    cfg = _bootstrap(args.root)
    runs_dir = cfg.paths.runs_dir
    if args.run_id:
        candidates = [runs_dir / args.run_id]
    else:
        candidates = sorted((p for p in runs_dir.glob("*") if (p / "state.json").is_file()),
                            key=lambda p: (p / "state.json").stat().st_mtime, reverse=True)[:1]
    if not candidates or not (candidates[0] / "state.json").is_file():
        print(f"No runs found under {runs_dir}")
        return 1
    state = RunState.load_or_create(candidates[0])
    print(state.render_table())
    return 0


def cmd_diagnose(args) -> int:
    """Full native-dependency report. Paste the output when a DLL fails to load."""
    import subprocess
    cfg = _bootstrap(args.root)
    script = cfg.root / "scripts" / "diagnose_mediapipe.py"
    return subprocess.run([sys.executable, str(script)], cwd=cfg.root).returncode


def cmd_selftest(args) -> int:
    """Run the synthetic race walker test suite. No video and no model needed."""
    import subprocess
    cfg = _bootstrap(args.root)
    cmd = [sys.executable, "-m", "pytest", str(cfg.root / "tests"), "-q"]
    return subprocess.run(cmd, cwd=cfg.root).returncode


def cmd_steps(args) -> int:
    for s in STEP_ORDER:
        print(f"{s:<4} {STEP_NAMES[s]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="racewalk", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="Project root. Defaults to the repository containing this package.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="Check the environment.")
    d.add_argument("--llm", action="store_true",
                   help="Also make one small structured-output call to the configured Ollama model.")
    d.set_defaults(func=cmd_doctor)

    i = sub.add_parser("init-session", help="Write a session.json template for a video.")
    i.add_argument("--video", required=True)
    i.add_argument("--session")
    i.add_argument("--athlete")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init_session)

    r = sub.add_parser("run", help="Run the pipeline.")
    r.add_argument("--video", required=True)
    r.add_argument("--session", help="Path to session.json. Defaults to sessions/<video stem>.json")
    r.add_argument("--run-id")
    r.add_argument("--from", dest="from_step", choices=STEP_ORDER)
    r.add_argument("--to", dest="to_step", choices=STEP_ORDER)
    r.add_argument("--only", choices=STEP_ORDER, help="Run exactly one step.")
    r.add_argument("--force", action="store_true", help="Ignore the resume cache.")
    r.add_argument("--no-graph", action="store_true", help="Run steps directly, bypassing LangGraph.")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="Show the state of a run.")
    s.add_argument("run_id", nargs="?")
    s.set_defaults(func=cmd_status)

    g = sub.add_parser("diagnose", help="Full native-dependency report for DLL failures.")
    g.set_defaults(func=cmd_diagnose)

    t = sub.add_parser("selftest", help="Run the synthetic race walker tests. No video needed.")
    t.set_defaults(func=cmd_selftest)

    l = sub.add_parser("steps", help="List the pipeline steps.")
    l.set_defaults(func=cmd_steps)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
