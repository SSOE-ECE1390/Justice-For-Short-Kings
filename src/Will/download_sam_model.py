"""
Script to download SAM (Segment Anything Model) checkpoints.
"""

import urllib.request
import ssl
import sys
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar:
    """Progress bar for downloads."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_sam_model(model_type="vit_b"):
    """
    Download SAM model checkpoint.

    Args:
        model_type: One of "vit_h" (huge, 2.4GB), "vit_l" (large, 1.2GB), "vit_b" (base, 375MB)
    """
    # SAM model URLs
    sam_models = {
        'vit_h': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'filename': 'sam_vit_h_4b8939.pth',
            'size': '2.4 GB'
        },
        'vit_l': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'filename': 'sam_vit_l_0b3195.pth',
            'size': '1.2 GB'
        },
        'vit_b': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'filename': 'sam_vit_b_01ec64.pth',
            'size': '375 MB'
        },
    }

    if model_type not in sam_models:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available models: {', '.join(sam_models.keys())}")
        return False

    model_info = sam_models[model_type]

    # Create assets directory
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    model_path = assets_dir / model_info['filename']

    # Check if already downloaded
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
        overwrite = input("Do you want to re-download? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Skipping download.")
            return True

    # Download
    print(f"\nDownloading SAM model: {model_type}")
    print(f"URL: {model_info['url']}")
    print(f"Size: {model_info['size']}")
    print(f"Destination: {model_path}")
    print("\nThis may take a while depending on your internet connection...")

    try:
        # Create SSL context that doesn't verify certificates (for macOS)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create opener with SSL context
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(
            model_info['url'],
            model_path,
            DownloadProgressBar()
        )
        print(f"\n✓ Download completed successfully!")
        print(f"Model saved to: {model_path}")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTrying alternative download method with curl...")

        # Fallback to curl
        import subprocess
        try:
            subprocess.run(
                ['curl', '-L', '-o', str(model_path), model_info['url']],
                check=True
            )
            print(f"\n✓ Download completed successfully with curl!")
            print(f"Model saved to: {model_path}")
            return True
        except Exception as curl_error:
            print(f"✗ Curl download also failed: {curl_error}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("SAM Model Downloader")
    print("=" * 60)

    print("\nAvailable SAM models:")
    print("  1. vit_b (Base)   - 375 MB  - Fastest, good quality")
    print("  2. vit_l (Large)  - 1.2 GB  - Better quality")
    print("  3. vit_h (Huge)   - 2.4 GB  - Best quality, slower")

    print("\nRecommendation:")
    print("  - For testing/quick results: vit_b")
    print("  - For production/best results: vit_h")

    # Get user choice
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        choice = input("\nEnter your choice (1-3) or model type: ").strip()

        if choice == "1":
            model_type = "vit_b"
        elif choice == "2":
            model_type = "vit_l"
        elif choice == "3":
            model_type = "vit_h"
        else:
            model_type = choice

    # Download
    success = download_sam_model(model_type)

    if success:
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nYou can now run the demo:")
        print("  python src/Will/demo.py")
        print("\nOr use the module in your code:")
        print("  from Will import SegmentationExpander")
    else:
        print("\n" + "=" * 60)
        print("Setup Failed!")
        print("=" * 60)
        print("\nYou can download manually from:")
        print("  https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print(f"\nPlace the .pth file in: {Path(__file__).parent / 'assets'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
