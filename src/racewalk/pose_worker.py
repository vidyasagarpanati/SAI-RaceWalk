"""Isolated MediaPipe pose worker.

Runs as its own process:  python -m racewalk.pose_worker <args>

WHY THIS EXISTS. On Windows, MediaPipe's native bindings fail with
    ImportError: DLL load failed while importing _framework_bindings
when pyarrow (which pandas 2.x imports eagerly) is already loaded in the same
process. Both ship copies of protobuf and abseil, and whichever DLL initialises
second loses. A bare `python -c "from mediapipe.tasks.python import vision"`
works while the same import inside the pipeline fails, which is exactly this.

So MediaPipe never shares a process with pandas, pyarrow or langgraph. This
module may import ONLY the standard library, numpy, cv2 and mediapipe. A test
enforces that. It writes a single .npz; the parent process builds the parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _selftest() -> int:
    from mediapipe.tasks.python import vision  # noqa: F401
    import mediapipe as mp
    import cv2
    import numpy as np
    loaded = sorted(m for m in ("pyarrow", "pandas", "langgraph") if m in sys.modules)
    print(json.dumps({
        "ok": True, "mediapipe": getattr(mp, "__version__", "?"),
        "cv2": cv2.__version__, "numpy": np.__version__,
        "conflicting_modules_loaded": loaded,
    }))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="racewalk.pose_worker")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frames-dir")
    ap.add_argument("--model")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--out")
    ap.add_argument("--options", default="{}", help="JSON of PoseLandmarker thresholds")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = json.loads(args.options)
    frames = sorted(Path(args.frames_dir).glob("f*.jpg"))
    n = len(frames)
    if n == 0:
        print("pose_worker: no frames found", file=sys.stderr)
        return 3

    delegate = getattr(mp_python.BaseOptions.Delegate, str(opts.get("delegate", "CPU")).upper())
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model, delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=int(opts.get("num_poses", 1)),
        min_pose_detection_confidence=float(opts.get("min_pose_detection_confidence", 0.5)),
        min_pose_presence_confidence=float(opts.get("min_pose_presence_confidence", 0.5)),
        min_tracking_confidence=float(opts.get("min_pose_tracking_confidence", 0.5)),
        output_segmentation_masks=False,
    )

    arr = np.full((n, 33, 8), np.nan, dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    unreadable = []
    t0 = time.time()
    report_every = max(1, n // 20)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, path in enumerate(frames):
            bgr = cv2.imread(str(path))
            if bgr is None:
                unreadable.append(path.name)
                continue
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(round(i * 1000.0 / args.fps)))
            if result.pose_landmarks:
                detected[i] = True
                lms = result.pose_landmarks[0]
                worlds = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                for j in range(33):
                    lm = lms[j]
                    arr[i, j, 0:5] = (lm.x, lm.y, lm.z,
                                      getattr(lm, "visibility", np.nan),
                                      getattr(lm, "presence", np.nan))
                    if worlds is not None:
                        w = worlds[j]
                        arr[i, j, 5:8] = (w.x, w.y, w.z)
            if (i + 1) % report_every == 0 or i + 1 == n:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  pose {i + 1:>6}/{n}  {100 * (i + 1) / n:5.1f}%  "
                      f"{rate:5.1f} fps  eta {(n - i - 1) / max(rate, 1e-6):5.0f}s", flush=True)

    np.savez_compressed(args.out, landmarks=arr, detected=detected,
                        unreadable=np.array(unreadable, dtype=object),
                        frame_names=np.array([p.name for p in frames], dtype=object))
    print(f"  pose done in {time.time() - t0:.1f}s, detected in {detected.mean():.1%} of frames",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
