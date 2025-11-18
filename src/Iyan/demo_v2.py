#!/usr/bin/env python3
"""
Demo and test utilities for the real-time height equalizer.
"""

from pathlib import Path
import sys
import cv2
import numpy as np


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    required = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_camera(camera_index=0):
    """Test if camera is accessible."""
    print(f"\nTesting camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"✗ Camera {camera_index} not accessible")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"✗ Cannot read from camera {camera_index}")
        return False
    
    print(f"✓ Camera {camera_index} working ({frame.shape[1]}x{frame.shape[0]})")
    return True


def find_model():
    """Try to locate the MediaPipe model file."""
    print("\nSearching for MediaPipe model...")
    
    search_paths = [
        Path.cwd(),
        Path.cwd() / "Iyan" / "assets",  # Check Iyan/assets
        Path.cwd() / "src" / "Iyan" / "assets",  # Check src/Iyan/assets
        Path(__file__).parent / "assets",  # Check same dir as script
        Path(__file__).parent,  # Check same dir as script
        Path.home() / "Downloads",
    ]
    
    model_names = [
        "pose_landmarker_full.task",
        "pose_landmarker_lite.task",
        "pose_landmarker_heavy.task",
    ]
    
    for search_path in search_paths:
        for model_name in model_names:
            model_path = search_path / model_name
            if model_path.exists():
                print(f"✓ Found model: {model_path}")
                return model_path
    
    print("✗ No model found")
    print("\nDownload a model:")
    print("  Full:  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    print("  Lite:  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    return None


def test_hat_generation():
    """Test the procedural hat generation."""
    print("\nTesting hat generation...")
    
    try:
        from realtime_height_equalizer import RealtimeHeightEqualizer
        
        # Create dummy instance to test hat generation
        class HatTester:
            def __init__(self):
                self.hat_models = []
            
            def _generate_top_hat(self):
                return RealtimeHeightEqualizer._generate_top_hat(self)
            
            def _generate_party_hat(self):
                return RealtimeHeightEqualizer._generate_party_hat(self)
            
            def _generate_crown(self):
                return RealtimeHeightEqualizer._generate_crown(self)
        
        tester = HatTester()
        
        # Generate hats
        top_hat = tester._generate_top_hat()
        party_hat = tester._generate_party_hat()
        crown = tester._generate_crown()
        
        print(f"✓ Top hat: {top_hat.shape}")
        print(f"✓ Party hat: {party_hat.shape}")
        print(f"✓ Crown: {crown.shape}")
        
        # Save preview
        preview = np.zeros((250, 900, 3), dtype=np.uint8) + 255
        
        for i, (hat, name) in enumerate([
            (top_hat, "Top Hat"),
            (party_hat, "Party Hat"),
            (crown, "Crown")
        ]):
            x_offset = i * 300
            h, w = hat.shape[:2]
            
            # Scale to fit
            scale = min(200 / w, 200 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(hat, (new_w, new_h))
            
            # Position
            x_start = x_offset + (300 - new_w) // 2
            y_start = (250 - new_h) // 2
            
            # Blend
            rgb = resized[:, :, :3]
            alpha = resized[:, :, 3:4] / 255.0
            
            roi = preview[y_start:y_start+new_h, x_start:x_start+new_w]
            blended = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
            preview[y_start:y_start+new_h, x_start:x_start+new_w] = blended
            
            # Label
            cv2.putText(
                preview, name,
                (x_offset + 20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2
            )
        
        cv2.imwrite("hat_preview.png", preview)
        print("✓ Saved hat preview to: hat_preview.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Hat generation failed: {e}")
        return False


def create_test_video():
    """Create a simple test video with synthetic people."""
    print("\nCreating test video...")
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (640, 480))
        
        for frame_num in range(90):  # 3 seconds
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
            
            # Draw simple stick figures at different "depths"
            # Person 1: Close and tall
            x1, y1_base = 150, 400
            height1 = 200
            cv2.line(frame, (x1, y1_base), (x1, y1_base - height1), (0, 0, 255), 8)
            cv2.circle(frame, (x1, y1_base - height1 - 20), 20, (0, 0, 255), -1)
            
            # Person 2: Far and short
            x2, y2_base = 450, 380
            height2 = 120
            cv2.line(frame, (x2, y2_base), (x2, y2_base - height2), (255, 0, 0), 5)
            cv2.circle(frame, (x2, y2_base - height2 - 12), 12, (255, 0, 0), -1)
            
            # Add text
            cv2.putText(
                frame, f"Frame {frame_num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            
            out.write(frame)
        
        out.release()
        print("✓ Test video saved to: test_video.mp4")
        print("  (Note: This is a simple test - real pose detection needs actual people)")
        return True
        
    except Exception as e:
        print(f"✗ Video creation failed: {e}")
        return False


def run_quick_demo(model_path):
    """Run a quick demo with the default camera."""
    print("\nStarting quick demo...")
    print("Press 'q' to quit, 'h' for hat mode, 's' for stretch mode")
    
    try:
        from realtime_height_equalizer import RealtimeHeightEqualizer
        
        equalizer = RealtimeHeightEqualizer(
            pose_model_path=model_path,
            camera_index=0,
        )
        
        equalizer.run(method="hat", debug=True)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests and optionally start demo."""
    print("=" * 60)
    print("Real-Time Height Equalizer - Demo & Test Suite")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check camera
    camera_ok = check_camera(0)
    
    # Find model
    model_path = find_model()
    
    # Test hat generation
    hat_ok = test_hat_generation()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Dependencies: {'✓' if True else '✗'}")
    print(f"Camera:       {'✓' if camera_ok else '✗'}")
    print(f"Model:        {'✓' if model_path else '✗'}")
    print(f"Hats:         {'✓' if hat_ok else '✗'}")
    
    if not camera_ok or not model_path:
        print("\n⚠ Cannot run demo - missing requirements")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    response = input("Run live demo? (y/n): ")
    
    if response.lower() == 'y':
        run_quick_demo(model_path)
    else:
        print("\nSkipping demo. Run manually with:")
        print(f"  python realtime_height_equalizer.py --model {model_path}")


if __name__ == "__main__":
    main()