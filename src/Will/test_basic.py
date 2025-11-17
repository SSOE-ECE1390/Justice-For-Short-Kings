"""
Basic test script without GUI - just tests the core functionality.
"""

import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Will.segmentation_expander import SegmentationExpander


def test_basic_functionality():
    """Test basic segmentation without displaying windows."""
    print("=" * 60)
    print("Testing Segmentation Expansion (No GUI)")
    print("=" * 60)

    # Find SAM model
    assets_dir = Path(__file__).parent / "assets"
    sam_models = list(assets_dir.glob("sam_vit_*.pth"))

    if not sam_models:
        print("✗ No SAM model found. Please run:")
        print("  python src/Will/download_sam_model.py")
        return False

    sam_checkpoint = str(sam_models[0])
    model_type = "vit_b" if "vit_b" in sam_checkpoint else "vit_h" if "vit_h" in sam_checkpoint else "vit_l"

    print(f"\n✓ Found SAM model: {Path(sam_checkpoint).name}")
    print(f"✓ Model type: {model_type}")

    # Create test image
    print("\n Creating test image...")
    test_images_dir = Path(__file__).parent / "test_images"
    test_images_dir.mkdir(exist_ok=True)
    test_image_path = test_images_dir / "test_stick_figures.jpg"

    # Create simple test image if it doesn't exist
    if not test_image_path.exists():
        import numpy as np

        # Create blank canvas
        h, w = 600, 800
        image = np.ones((h, w, 3), dtype=np.uint8) * 255

        def draw_stick_figure(img, center_x, bottom_y, height, color=(0, 0, 0), thickness=3):
            head_radius = int(height * 0.12)
            torso_len = int(height * 0.35)
            leg_len = int(height * 0.4)
            arm_len = int(height * 0.3)

            head_y = bottom_y - height + head_radius
            cv2.circle(img, (center_x, head_y), head_radius, color, thickness)

            torso_top = head_y + head_radius
            torso_bottom = torso_top + torso_len
            cv2.line(img, (center_x, torso_top), (center_x, torso_bottom), color, thickness)

            arm_y = torso_top + torso_len // 3
            cv2.line(img, (center_x - arm_len, arm_y), (center_x + arm_len, arm_y), color, thickness)

            leg_left_x = center_x - leg_len // 2
            leg_right_x = center_x + leg_len // 2
            cv2.line(img, (center_x, torso_bottom), (leg_left_x, bottom_y), color, thickness)
            cv2.line(img, (center_x, torso_bottom), (leg_right_x, bottom_y), color, thickness)

        draw_stick_figure(image, 200, 550, 400, color=(255, 0, 0), thickness=4)
        draw_stick_figure(image, 400, 550, 250, color=(0, 255, 0), thickness=4)
        draw_stick_figure(image, 600, 550, 350, color=(0, 0, 255), thickness=4)

        cv2.imwrite(str(test_image_path), image)
        print(f"✓ Created test image: {test_image_path}")
    else:
        print(f"✓ Using existing test image: {test_image_path}")

    # Load image
    image = cv2.imread(str(test_image_path))
    print(f"✓ Image loaded: {image.shape}")

    # Initialize SegmentationExpander
    print("\n Initializing SegmentationExpander...")
    try:
        expander = SegmentationExpander(
            sam_checkpoint=sam_checkpoint,
            sam_model_type=model_type,
            device="cpu",
            use_mediapipe=False
        )
        print("✓ SegmentationExpander initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

    # Process image
    print("\n Processing image (this may take 10-30 seconds)...")
    try:
        results = expander.process_image(image, expansion_method="stretch", debug=True)
        print("\n✓ Image processed successfully!")

        # Save results
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(output_dir / "1_original.jpg"), results['original'])
        cv2.imwrite(str(output_dir / "2_masked.jpg"), results['masked'])
        cv2.imwrite(str(output_dir / "3_expanded_masked.jpg"), results['expanded_masked'])
        cv2.imwrite(str(output_dir / "4_final.jpg"), results['final'])

        print(f"\n✓ Results saved to: {output_dir}")
        print("  - 1_original.jpg")
        print("  - 2_masked.jpg")
        print("  - 3_expanded_masked.jpg")
        print("  - 4_final.jpg")

        return True

    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nTesting basic functionality without GUI...")
    success = test_basic_functionality()

    if success:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour segmentation expansion feature is working!")
        print("\nNext steps:")
        print("  1. Check the output images in src/Will/test_output/")
        print("  2. Run the full demo with GUI: python src/Will/demo.py")
        print("  3. Try with your own images!")
    else:
        print("\n" + "=" * 60)
        print("✗ TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
