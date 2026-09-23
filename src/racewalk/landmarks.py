"""MediaPipe Pose 33-landmark definitions, topology and drawing regions.

NOTE ON THE ORIGINAL PROMPT. A landmark table names IDs 19-22 as
LEFT_HAND / RIGHT_HAND / LEFT_FOOD_TIP / RIGHT_FOOD_TIP. Those are not the
names MediaPipe uses. The actual solution defines:

    19 LEFT_INDEX   20 RIGHT_INDEX   21 LEFT_THUMB   22 RIGHT_THUMB

The IDs and their anatomical positions are identical, only the labels differ,
so nothing about the geometry changes. ``PROMPT_ALIAS`` below keeps the
prompt's labels available for the report so the output still reads the way the
protocol specifies, while the code uses the real names.
"""
from __future__ import annotations

NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
assert len(NAMES) == 33

ID = {name: i for i, name in enumerate(NAMES)}

PROMPT_ALIAS = {
    19: "LEFT_HAND", 20: "RIGHT_HAND",
    21: "LEFT_FOOD_TIP", 22: "RIGHT_FOOD_TIP",
    9: "LEFT_MOUTH", 10: "RIGHT_MOUTH",
}


def report_name(idx: int) -> str:
    """The label the master prompt expects to see in the report."""
    return PROMPT_ALIAS.get(idx, NAMES[idx])


FACE = tuple(range(0, 11))          # red in the annotated frames
UPPER_LIMB = tuple(range(11, 23))   # green
LOWER_LIMB = tuple(range(23, 33))   # blue

REGION_OF = {}
for _i in FACE:
    REGION_OF[_i] = "face"
for _i in UPPER_LIMB:
    REGION_OF[_i] = "upper_limb"
for _i in LOWER_LIMB:
    REGION_OF[_i] = "lower_limb"

# BGR, because OpenCV. Matches the prompt: red face, green upper limb, blue lower limb.
REGION_COLOR_BGR = {
    "face": (60, 60, 235),
    "upper_limb": (90, 210, 90),
    "lower_limb": (235, 140, 60),
}

CONNECTIONS: tuple[tuple[int, int], ...] = (
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (0, 9), (0, 10),
    # shoulder girdle and arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (17, 19), (19, 21), (16, 18), (18, 20), (20, 22),
    # trunk
    (11, 23), (12, 24), (23, 24),
    # legs
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
)


def side(prefix: str) -> dict[str, int]:
    """Landmark ids for one side, keyed by joint. ``prefix`` is LEFT or RIGHT."""
    p = prefix.upper()
    return {
        "shoulder": ID[f"{p}_SHOULDER"],
        "elbow": ID[f"{p}_ELBOW"],
        "wrist": ID[f"{p}_WRIST"],
        "index": ID[f"{p}_INDEX"],
        "thumb": ID[f"{p}_THUMB"],
        "pinky": ID[f"{p}_PINKY"],
        "hip": ID[f"{p}_HIP"],
        "knee": ID[f"{p}_KNEE"],
        "ankle": ID[f"{p}_ANKLE"],
        "heel": ID[f"{p}_HEEL"],
        "foot": ID[f"{p}_FOOT_INDEX"],
        "ear": ID[f"{p}_EAR"],
        "eye": ID[f"{p}_EYE"],
    }


def leg_ids(prefix: str) -> dict[str, int]:
    """Landmark ids for one leg. ``prefix`` is LEFT or RIGHT."""
    q = side(prefix)
    return {k: q[k] for k in ("hip", "knee", "ankle", "heel", "foot")}


FOOT_LM = {"LEFT": ("LEFT_HEEL", "LEFT_FOOT_INDEX"), "RIGHT": ("RIGHT_HEEL", "RIGHT_FOOT_INDEX")}
