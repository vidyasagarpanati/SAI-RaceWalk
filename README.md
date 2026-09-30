# SAI RaceWalk: Race Walking Biomechanics Analysis Pipeline

Turns a recorded race walking video into two outputs, reproducibly:

1. an **annotated video** (skeleton, live joint angles, both legs' gait phase, foot-contact markers,
   straight-leg and event flags), plus short clips of each flagged event
2. a self-contained **HTML report** with embedded annotated screenshots, in the fixed nine-section
   format of your race walking analysis template

It is the sibling of the Archery pipeline and keeps its one rule:

> **Python measures. The model only writes prose over a frozen evidence file.**
> Any number that is not already in `05_metrics.json` cannot appear in the report.

## Pipeline

| Step | Name | Owner | Key output |
|---|---|---|---|
| S0 | ingest | ffprobe + schema | video hash, measured fps, validated session |
| S1 | frames | ffmpeg | one decode pass to JPEG |
| S2 | pose | MediaPipe | 33 landmarks per frame |
| S3 | kinematics | NumPy | joint angles, pelvis and trunk orientation, rotation, foot signals |
| S4 | gait | rules | contacts, heel strikes, toe-offs, strides, six phases per stride |
| S5 | stats | NumPy | **`05_metrics.json`**: evidence, flagged events, risk levels, technique status |
| S6 | annotate | OpenCV | annotated screenshots: flagged events plus one reference stride per leg |
| S7 | video | ffmpeg | annotated MP4, event clips, versioned copy in `outputs/` |
| S8 | narrate | Qwen via Ollama | eight JSON calls, one per written section |
| S9 | verify | guardrail | checklist, numeric grounding, camera-ceiling checks, repair loop |
| S10 | render | Jinja2 | one self-contained versioned HTML |

## What Python decides, and what the model writes

| Python (deterministic) | Model (prose, cites evidence keys) |
|---|---|
| every number, the event list, event severity and confidence | what each event means and a coaching cue |
| technique status per row, injury-risk level per category | explanation of those results |
| Sections 7 and 8 generic guidance (`config/guidance.yaml`) | training plan and targeted strength work |

## The camera decides how much can be claimed

Knee and hip flexion happen in the depth direction of a rear or front camera. From those views the
straight-leg and contact results are **screens**, and confidence for depth-derived leg angles is capped
at LOW (oblique MEDIUM, side HIGH). Foot strike and step length are reported only from a side view.
Pace and step length in metres are reported only if the session file has `treadmill_speed_kmh`, or
(overground/track footage) `distance_walked_m` — set at session creation with
`racewalk init-session --distance-m <metres>` — which gives average speed as distance over clip
duration. Neither is ever estimated from the picture. Average speed, once available, is shown live
alongside the joint angles in the video and screenshots. The report never rules on Rule 54: a judge
does.

In a side view (`side_left`/`side_right`), joint angles for the leg/arm facing away from the camera
(knee, hip, ankle, elbow) are shown dimmed in the live overlay and screenshots, not hidden: the value is
still measured, just not visually confirmable from that angle.

## Getting started (Windows, same box as Archery)

```powershell
.\scripts\bootstrap.ps1
racewalk init-session --video "D:\walks\athlete_01.mp4"
notepad sessions\athlete_01.json          # set camera_view (rear, front, side_left, side_right), speed if known
racewalk run --video "D:\walks\athlete_01.mp4" --to S5    # inspect the numbers first
racewalk run --video "D:\walks\athlete_01.mp4"
```

Docs: [`docs/02_RUNBOOK.md`](docs/02_RUNBOOK.md) for operation and tuning, [`docs/00_PLAN.md`](docs/00_PLAN.md) for design.

## Status

| Steps | State |
|---|---|
| S3 kinematics to S10 render | implemented, 13 tests pass on a synthetic walker with known ground truth |
| S0 ingest, S1 frames, S2 pose | carried over from Archery, not yet run on real race walking video |
| Thresholds in `config/phase_rules.yaml` | provisional, tune on real footage |
| `config/benchmarks.json` | empty on purpose, every reference value needs a citation |

`racewalk selftest` runs the whole measurement and report layer against the synthetic walker. No video and no model.
