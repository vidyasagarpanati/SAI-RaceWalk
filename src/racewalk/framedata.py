"""Loads per-frame landmark pixels and kinematics once, for S6, S7 and S10."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def frame_size(run_dir: Path) -> tuple[int, int]:
    """(width, height) of the DECODED frames. Not the probe values: a phone clip with
    rotation metadata is decoded upright, so only the frames know the truth."""
    fm = json.loads((run_dir / "01_frames.json").read_text(encoding="utf-8"))
    if fm.get("width") and fm.get("height"):
        return int(fm["width"]), int(fm["height"])
    frames = sorted(Path(fm["frames_dir"]).glob("f*.jpg"))
    if frames:
        img = cv2.imread(str(frames[0]))
        if img is not None:
            return int(img.shape[1]), int(img.shape[0])
    ing = run_dir / "00_ingest.json"
    if ing.is_file():
        pr = json.loads(ing.read_text(encoding="utf-8")).get("probe", {})
        if pr.get("width") and pr.get("height"):
            return int(pr["width"]), int(pr["height"])
    return 1920, 1080


@dataclass
class FrameData:
    frames: list[Path]
    fps: float
    width: int
    height: int
    pts: np.ndarray          # (n, 33, 2) pixels
    vis: np.ndarray          # (n, 33)
    kin: pd.DataFrame
    gait: dict
    metrics: dict | None
    fh: pd.DataFrame | None = None
    _phases: dict | None = None

    @classmethod
    def load(cls, run_dir: Path, with_metrics: bool = True) -> "FrameData":
        fm = json.loads((run_dir / "01_frames.json").read_text(encoding="utf-8"))
        frames = sorted(Path(fm["frames_dir"]).glob("f*.jpg"))
        w, h = frame_size(run_dir)
        lm = pd.read_parquet(run_dir / "02_landmarks.parquet")
        n = int(lm["frame"].max()) + 1
        xy = lm[["x", "y"]].to_numpy(float).reshape(n, 33, 2) * np.array([w, h])
        vis = lm["visibility"].to_numpy(float).reshape(n, 33)
        kin = pd.read_parquet(run_dir / "03_kinematics.parquet")
        gait = json.loads((run_dir / "04_gait.json").read_text(encoding="utf-8"))
        metrics = None
        mp = run_dir / "05_metrics.json"
        if with_metrics and mp.is_file():
            metrics = json.loads(mp.read_text(encoding="utf-8"))
        fhp = run_dir / "04_foot_height.parquet"
        fh = pd.read_parquet(fhp) if fhp.is_file() else None
        return cls(frames, float(fm["analysis_fps"]), w, h, xy, vis, kin, gait, metrics, fh)

    def contact(self, leg: str, frame: int):
        if self.fh is None:
            return None
        return bool(self.fh[f"contact_{leg}"].iloc[frame])

    def foot_height(self, leg: str, frame: int):
        if self.fh is None:
            return None
        v = float(self.fh[f"height_{leg}"].iloc[frame])
        return v if np.isfinite(v) else None

    def legs_at(self, frame: int) -> dict:
        if self._phases is None:
            self._phases = self.phase_of_frame()
        return {lg: {"phase": self._phases[lg][frame][0], "stride_id": self._phases[lg][frame][1],
                     "contact": self.contact(lg, frame)} for lg in ("L", "R")}

    def leg_length_px(self) -> float:
        from racewalk.landmarks import ID
        tot = []
        for p in ("LEFT", "RIGHT"):
            hip, kn, an = self.pts[:, ID[f"{p}_HIP"]], self.pts[:, ID[f"{p}_KNEE"]], self.pts[:, ID[f"{p}_ANKLE"]]
            tot.append(np.linalg.norm(hip - kn, axis=1) + np.linalg.norm(kn - an, axis=1))
        return float(np.nanmedian(np.concatenate(tot)))

    def shoulder_width_px(self) -> float:
        from racewalk.landmarks import ID
        d = np.linalg.norm(self.pts[:, ID["LEFT_SHOULDER"]] - self.pts[:, ID["RIGHT_SHOULDER"]], axis=1)
        return float(np.nanmedian(d))

    def phase_of_frame(self) -> dict[str, list]:
        """Per leg and per frame: (phase code, stride id). Both legs are always
        in some phase at once, so the video can show both."""
        n = len(self.kin)
        out = {"L": [(None, None)] * n, "R": [(None, None)] * n}
        for leg in ("L", "R"):
            arr: list = [(None, None)] * n
            for st in self.gait["strides"]:
                if st["leg"] != leg:
                    continue
                for p in st["phases"]:
                    if p["detected"]:
                        for i in range(p["start_frame"], min(n, p["end_frame"] + 1)):
                            arr[i] = (p["phase"], st["id"])
            out[leg] = arr
        return out
