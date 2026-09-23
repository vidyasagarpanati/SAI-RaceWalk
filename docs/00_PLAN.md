# SAI RaceWalk: design notes

Ported from the Archery pipeline. The step contracts, evidence grounding, resume cache, write whitelist and
versioned outputs are unchanged. What changed is the sport model.

## From shots to strides

Archery has a few long shots with nine phases each. Race walking has dozens of short strides. A **stride** is one
leg's heel strike to its next heel strike. Each is split into six contiguous phases: initial contact, loading
(straight-leg window), mid-stance (vertical), push-off, early swing, late swing. Both legs are always in some phase
at once, so the overlay and video show both.

Gait events come from foot height above each foot's own ground level, in leg lengths, with hysteresis thresholds set
as fractions of that foot's swing amplitude. The vertical (mid-stance) moment is when the opposite foot is highest,
which works from behind. Flight candidates use the raw hysteresis states, because cleaning would merge exactly the
short episodes the screen exists to find.

## Evidence keys

`L.<PHASE>.<measure>.<stat>`, `R....`, `both.stride.<metric>.<stat>`, `asym.<metric>.pct`,
`early.` / `late.` / `drift.<metric>...`, `gait.*`, `event<N>.*`, `risk.<category>.*`, `screen.*`, `session.*`,
`quality.*`. Per-stride rows are in `05_strides.json`; only aggregates are evidence.

## Report

Nine sections mirror the analysis template: timestamps of interest, technique breakdown, injury risk, comparison to
ideal form, speed and efficiency, coaching and training plan, strength and conditioning, sports science, limitations.
Eight are model calls; Section 8 is static guidance from `config/guidance.yaml`. The report shows event frames plus one
reference stride per leg. The video shows everything.

## Decisions

| # | Decision |
|---|---|
| R1 | Status, severity, confidence and risk level are computed in Python. The model explains, never decides. |
| R2 | Confidence is capped by camera view (frontal LOW, oblique MEDIUM, sagittal HIGH for depth-derived leg angles). |
| R3 | Pace and step length in metres only from a supplied belt speed. Never estimated from the picture. |
| R4 | No reference values without a citation. Benchmarks ship empty. |
| R5 | Risk thresholds are labelled provisional screening heuristics, printed beside every level. |
| R6 | The model may not type degrees, percent, cadence or speed in training fields (local check). |
| R7 | Annotated video is downscaled to 1920 px wide before drawing, so a 4K clip costs 1080p-class time. |

## Known limits

Real footage will need threshold tuning. Contact and straight leg are screens, not rulings. Rotation comes from
estimated depth and is capped at MEDIUM in every view. The synthetic test walker proves the logic, not the accuracy
of MediaPipe on your athletes.
