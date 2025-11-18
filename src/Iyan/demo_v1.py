from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .height_equalizer import HeightEqualizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Justice-for-Short-Kings – static image demo.\n"
            "1) Detect everyone in the photo.\n"
            "2) Find the depth-aware shortest and tallest person.\n"
            "3) Either stretch the shortest or give them a height-equalizing hat."
        )
    )
    parser.add_argument("input", type=Path, help="Input image path.")
    parser.add_argument("output", type=Path, help="Output image path.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="MediaPipe pose_landmarker_*.task model file.",
    )
    parser.add_argument(
        "--method",
        choices=["stretch", "accessory", "hat", "both"],
        default="stretch",
        help="How to help the shortest king.",
    )
    parser.add_argument(
        "--accessory",
        type=Path,
        default=None,
        help="Optional RGBA PNG for a custom hat/accessory.",
    )
    parser.add_argument(
        "--shortest-index",
        type=int,
        default=None,
        help=(
            "Override automatic shortest-person detection "
            "and force this person index (0-based) to be modified."
        ),
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help=(
            "Override automatic tallest-person detection "
            "and use this person index (0-based) as the reference."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.input}")

    equalizer = HeightEqualizer(
        pose_model_path=args.model,
        accessory_path=args.accessory,
    )

    try:
        result = equalizer.equalize(
            image,
            method=args.method,
            shortest_index=args.shortest_index,
            reference_index=args.reference_index,
        )
    finally:
        equalizer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result.output_image)

    if not result.measurements:
        print("No people detected – saved original image with no changes.")
        return

    print(f"Detected {len(result.measurements)} person(s).")
    print(
        f"Shortest index: {result.shortest_index}, "
        f"tallest index: {result.reference_index}"
    )

    if result.applied:
        print(
            f"Applied '{result.method}' to person #{result.shortest_index} "
            f"to match person #{result.reference_index}. "
            f"Output saved to {args.output}."
        )
    else:
        print(
            "Heights already comparable or indices identical – "
            f"no visual change applied. Saved {args.output}."
        )


if __name__ == "__main__":
    main()
