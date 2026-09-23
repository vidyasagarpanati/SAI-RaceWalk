"""Gait phase names, badge numbers and colours, plus the flagged-event vocabulary.

A stride is one leg's cycle from heel strike to the next heel strike of the same leg.
It is split into six contiguous phases. The straight-leg rule (World Athletics Rule 54.2)
applies from initial contact until the leg passes the vertical, which is the union of
INITIAL_CONTACT, LOADING and the first half of MID_STANCE in this segmentation.
"""
from __future__ import annotations

# (code, badge number, display name, BGR colour for OpenCV overlays)
PHASES: list[tuple[str, int, str, tuple[int, int, int]]] = [
    ("INITIAL_CONTACT", 1, "Initial contact (heel strike)", (60, 60, 220)),   # red
    ("LOADING", 2, "Loading (straight-leg window)", (0, 140, 255)),           # orange
    ("MID_STANCE", 3, "Mid-stance (vertical)", (60, 170, 60)),                # green
    ("PUSH_OFF", 4, "Push-off / toe-off", (200, 110, 30)),                    # blue
    ("EARLY_SWING", 5, "Early swing", (150, 60, 130)),                        # purple
    ("LATE_SWING", 6, "Late swing (reach)", (128, 128, 128)),                 # grey
]

ORDER = [p[0] for p in PHASES]
BADGE = {p[0]: p[1] for p in PHASES}
DISPLAY = {p[0]: p[2] for p in PHASES}
COLOR_BGR = {p[0]: p[3] for p in PHASES}

# The phases in which the advancing leg must be straight under Rule 54.2.
STRAIGHT_LEG_PHASES = ["INITIAL_CONTACT", "LOADING", "MID_STANCE"]

# Flagged event types (Section 1, "Timestamps of Interest").
# code -> (display name, BGR colour)
EVENTS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "STRAIGHT_LEG": ("Straight-leg concern", (50, 50, 235)),
    "CONTACT": ("Contact ambiguity (both feet airborne)", (0, 140, 255)),
    "ASYMMETRY": ("Left-right asymmetry", (0, 215, 255)),
    "FATIGUE": ("Fatigue / form drift", (200, 90, 20)),
}
EVENT_ORDER = list(EVENTS)
EVENT_DISPLAY = {k: v[0] for k, v in EVENTS.items()}
EVENT_COLOR_BGR = {k: v[1] for k, v in EVENTS.items()}

LEGS = ("L", "R")
LEG_NAME = {"L": "Left", "R": "Right"}
LEG_PREFIX = {"L": "LEFT", "R": "RIGHT"}


# What to look for in each phase. Static reference text, not model output and not a
# measurement. It fills the coaching box on the reference-stride screenshots.
PHASE_LOOK_FOR = {
    "INITIAL_CONTACT": "Heel lands ahead of the body with the knee extended. Check the knee reads straight at the moment the foot touches.",
    "LOADING": "Straight-leg window. The knee must not flex while the body travels over the planted foot.",
    "MID_STANCE": "Leg passes the vertical. This is the last moment the rule requires the knee to be straight. Check pelvis level and trunk upright.",
    "PUSH_OFF": "Knee may now bend. Look for a full, balanced drive from the ankle and a pelvis that stays level.",
    "EARLY_SWING": "Swing foot travels low. High clearance costs efficiency and risks a visible loss of contact.",
    "LATE_SWING": "Leg reaches forward and extends before contact. Look for the knee straightening early, not at the last instant.",
}
