from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .height_equalizer import HeightEqualizer, AccessoryStyle


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Equalize the height of the shortest person in a photo so they match "
            "another person."
        )
    )
    parser.add_argument("image", type=Path, help="Path to the input photo.")
    parser.add_argument(
        "output", type=Path, help="Where to store the adjusted image output."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the MediaPipe pose landmarker .task file.",
    )
    parser.add_argument(
        "--method",
        choices=("stretch", "accessory", "hat", "both"),
        default="stretch",
        help=(
            "How to equalize the shortest person. "
            "'stretch' scales them up, "
            "'accessory' / 'hat' adds an accessory, "
            "'both' does a small stretch plus an accessory."
        ),
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help="Optional index of the person whose height should be matched.",
    )
    parser.add_argument(
        "--shortest-index",
        type=int,
        default=None,
        help="Optional override for who should be treated as the shortest person.",
    )
    parser.add_argument(
        "--accessory",
        type=Path,
        default=None,
        help="Optional path to a transparent PNG accessory for the accessory method.",
    )
    parser.add_argument(
        "--accessory-style",
        type=str,
        choices=[s.value for s in AccessoryStyle],
        default=AccessoryStyle.TOP_HAT.value,
        help=(
            "Built-in accessory style to use when no custom PNG is given. "
            "Choices: " + ", ".join(s.value for s in AccessoryStyle)
        ),
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {args.image}")

    equalizer = HeightEqualizer(
        pose_model_path=args.model,
        accessory_path=args.accessory,
        accessory_style=args.accessory_style,
    )

    try:
        result = equalizer.equalize(
            image,
            method=args.method,
            reference_index=args.reference_index,
            shortest_index=args.shortest_index,
        )
    finally:
        equalizer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result.output_image)

    if result.applied:
        print(
            f"Equalized person #{result.shortest_index} to match #{result.reference_index} "
            f"using '{result.method}'. Saved to {args.output}."
        )
    else:
        print("All detected people already have comparable heights. No changes made.")


if __name__ == "__main__":
    main()
