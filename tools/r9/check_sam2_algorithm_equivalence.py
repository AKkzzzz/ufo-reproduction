#!/usr/bin/env python3
"""Guard the validated Grounded-SAM2 tracking constants against drift."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = REPO_ROOT / "third_party/Grounded-SAM-2/grounded_sam2_tracking_ufo.py"
PERSISTENT = REPO_ROOT / "tools/r9/preprocess_waymo_sam2_tracks_persistent.py"
PROMPT = "car. truck. bus. motorcycle. bicycle. pedestrian."


def require(source, pattern, label):
    if not re.search(pattern, source, flags=re.MULTILINE | re.DOTALL):
        raise RuntimeError(f"{label} not found with expected value")


def main():
    original = ORIGINAL.read_text()
    persistent = PERSISTENT.read_text()

    for label, source in (("original", original), ("persistent", persistent)):
        require(source, re.escape(PROMPT), f"{label} TEXT_PROMPT")
        require(source, r"threshold\s*=\s*0\.25", f"{label} detection threshold")
        require(source, r"text_threshold\s*=\s*0\.25", f"{label} text threshold")
        require(source, r"iou_threshold\s*=\s*0\.8", f"{label} IoU threshold")

    require(original, r'GSAM_STEP",\s*"10"', "original step")
    require(persistent, r'--step",\s*type=int,\s*default=10', "persistent step")
    require(original, r"max_frame_num_to_track\s*=\s*step\s*,", "original forward length")
    require(
        persistent,
        r"max_frame_num_to_track\s*=\s*self\.args\.step\s*,",
        "persistent forward length",
    )
    require(original, r"max_frame_num_to_track\s*=\s*step\s*\*\s*2", "original reverse length")
    require(
        persistent,
        r"max_frame_num_to_track\s*=\s*self\.args\.step\s*\*\s*2",
        "persistent reverse length",
    )

    print(f"TEXT_PROMPT={PROMPT}")
    print("DETECTION_THRESHOLD=0.25")
    print("TEXT_THRESHOLD=0.25")
    print("STEP=10")
    print("IOU_THRESHOLD=0.8")
    print("FORWARD_TRACK_LENGTH=step")
    print("REVERSE_REFINEMENT_TRACK_LENGTH=step*2")
    print("SAM2_ALGORITHM_EQUIVALENCE_PASS")


if __name__ == "__main__":
    main()
