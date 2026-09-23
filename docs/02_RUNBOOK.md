# Runbook

Target: the same remote Windows box as Archery (RTX 3060, Ollama `qwen3.6:35b`). Everything below is PowerShell.

## One-time setup

```powershell
git clone <your repo> SAI-RaceWalk ; cd SAI-RaceWalk
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\bootstrap.ps1
ollama list                     # confirm the tag, then set llm.model in config\pipeline.yaml
racewalk doctor --llm
```

## Per video

1. `racewalk init-session --video "D:\walks\athlete_01.mp4"`
2. Edit `sessions\athlete_01.json`. **`camera_view` is mandatory and changes what can be claimed.**
   Add `treadmill_speed_kmh` if you know it (pace and step length depend on it), `body_mass_kg` to scale
   the generic nutrition line, and `level`. Anything left null is printed as NOT PROVIDED, never assumed.
3. `racewalk run --video ... --to S5` and read the numbers before spending GPU time.
4. `racewalk run --video ...` for the annotated video and the report.

Outputs: `outputs\RaceWalk_Report_<athlete>_<date>_vNN.html` and `outputs\<Athlete>_Annotated_<date>_vNN.mp4`.
Versions increment. Nothing is overwritten. Per-event clips are in `runs\<run_id>\07_video`.

## Tuning gait detection (S4)

Thresholds live in `config/phase_rules.yaml`. Tune from data, not by eye:

1. Run to S5. Open `runs\<run_id>\04_gait.json` and read `signal_summary` and each leg's
   `amplitude_leg_lengths`, `enter_leg_lengths`, `exit_leg_lengths`.
2. If few strides are found, the foot signal is weak: check that heel and toe landmarks track, then adjust
   `contact.enter_frac_of_amplitude` and `exit_frac_of_amplitude`.
3. Compare detected heel strikes with the video in `07_video`. The detector fires slightly before the true
   strike on a shallow foot approach; `screening.loading_window_start_after_hs_s` skips that margin.
4. Rerun with `--from S4`. Pose estimation is not repeated.

Screening thresholds (`straight_knee_min_deg`, asymmetry, drift, flight) and the risk levels in
`config/risk_rules.yaml` are **provisional defaults, not literature values**. Set them with your coach.

## Frame rate

Judges see contact loss at speeds a 60 fps clip cannot resolve. At 60 fps a contact flag is a hint, and its
confidence is capped at LOW (MEDIUM only from a side view at 120 fps or more). Film side views at 120 fps or higher.

## Reference values

`config/benchmarks.json` ships empty. To add one you need a citation. An entry without `source`, or with
status `NEEDS_SOURCE`, is rejected and listed in the report. The pelvic rotation range in the original
hand-made sample report has no citation, so it is not used.

## Day-to-day commands

| Need | Command |
|---|---|
| State of the last run | `racewalk status` |
| One step only | `racewalk run --video ... --only S4` |
| Redo the narrative after editing a prompt | `racewalk run --video ... --from S8 --force` |
| Ignore the resume cache | `racewalk run --video ... --force` |
| Run the synthetic tests | `racewalk selftest` |

A section the model cannot produce after all attempts is rendered NOT AVAILABLE and the file is published
`..._PARTIAL_vNN.html`. Set `report.allow_partial: false` to require a complete report.

## Guardrails

The source video is opened read-only; there is no delete primitive; all writes go through `io_guard`;
reports are versioned; S9 blocks any number not in `05_metrics.json`; injury language is screened against
`config/banned_phrases.txt`; Ollama is called on localhost only.
