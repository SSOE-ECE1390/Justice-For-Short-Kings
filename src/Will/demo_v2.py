"""
Demo script for 2-Stage Segmentation Expansion (YOLO + SAM)
"""

import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Will.segmentation_expander_v2 import SegmentationExpanderV2


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python demo_v2.py <image_path>")
        print("\nExample:")
        print("  python src/Will/demo_v2.py src/Will/test_images/test_image1.png")
        sys.exit(1)

    image_path = sys.argv[1]

    # Find SAM model
    assets_dir = Path(__file__).parent / "assets"
    sam_models = list(assets_dir.glob("sam_vit_*.pth"))

    if not sam_models:
        print("Error: No SAM model found!")
        print("Please run: python src/Will/download_sam_model.py")
        sys.exit(1)

    sam_checkpoint = str(sam_models[0])
    model_type = "vit_b" if "vit_b" in sam_checkpoint.lower() else \
                 "vit_l" if "vit_l" in sam_checkpoint.lower() else "vit_h"

    # Load image
    print("=" * 60)
    print("2-STAGE SEGMENTATION EXPANSION DEMO")
    print("Using YOLO + SAM for Precise Person Segmentation")
    print("=" * 60)
    print(f"\nLoading image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)

    print(f"✓ Image loaded: {image.shape}")
    print(f"✓ SAM model: {Path(sam_checkpoint).name}")
    print(f"✓ SAM type: {model_type}")
    print(f"✓ YOLO model: yolov8n.pt (nano)")

    # Initialize system
    print("\nInitializing models...")
    expander = SegmentationExpanderV2(
        sam_checkpoint=sam_checkpoint,
        sam_model_type=model_type,
        yolo_model="yolov8n.pt",  # Fastest YOLO model
        device="cpu"
    )

    # Process image
    print("\n" + "=" * 60)
    print("STARTING PROCESSING PIPELINE")
    print("=" * 60)

    results = expander.process_image(image, conf_threshold=0.5)

    # Display results
    expander.display_results(results, max_height=900)

    print("\n" + "=" * 60)
    print("✓ DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
