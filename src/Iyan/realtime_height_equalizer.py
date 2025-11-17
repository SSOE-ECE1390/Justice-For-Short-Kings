from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ArrayF32 = npt.NDArray[np.float32]
ArrayU8 = npt.NDArray[np.uint8]


@dataclass
class Person3D:
    """Person with 3D depth-adjusted measurements."""
    
    index: int
    landmarks_2d: ArrayF32  # (33, 2) pixel coordinates
    landmarks_3d: ArrayF32  # (33, 3) with depth
    visibility: ArrayF32  # (33,)
    height_pixels: float
    depth_z: float  # Average depth from camera
    real_height_estimate: float  # Depth-corrected height
    head_top_y: int
    head_center_x: int
    head_width: float
    shoulder_width: float


class RealtimeHeightEqualizer:
    """Real-time video processor that equalizes heights with depth awareness."""
    
    def __init__(
        self,
        pose_model_path: Path | str,
        *,
        hat_model_path: Optional[Path | str] = None,
        camera_index: int = 0,
        max_people: int = 5,
        min_confidence: float = 0.5,
    ):
        self.pose_model_path = Path(pose_model_path)
        if not self.pose_model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.pose_model_path}")
        
        # Initialize MediaPipe Pose for VIDEO mode
        base_options = mp_python.BaseOptions(model_asset_path=str(self.pose_model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            output_segmentation_masks=True,
        )
        
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.camera_index = camera_index
        self.cap = None
        
        # Load or generate 3D hat models
        self.hat_models = self._load_hat_models(hat_model_path)
        
        # Tracking
        self.frame_count = 0
        self.total_frame_count = 0  # Never reset - for timestamps
        self.fps = 0
        self.last_time = time.time()
        
    def _load_hat_models(self, hat_path: Optional[Path | str]) -> List[ArrayU8]:
        """Load multiple hat designs as RGBA images."""
        hats = []
        
        if hat_path and Path(hat_path).exists():
            hat = cv2.imread(str(hat_path), cv2.IMREAD_UNCHANGED)
            if hat is not None and hat.shape[2] == 4:
                hats.append(hat)
        
        # Generate default hats if none loaded
        if not hats:
            hats.extend([
                self._generate_top_hat(),
                self._generate_party_hat(),
                self._generate_crown(),
            ])
        
        return hats
    
    def _generate_top_hat(self) -> ArrayU8:
        """Generate a classic top hat."""
        w, h = 300, 200
        hat = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Crown (tall part)
        crown_w = int(w * 0.5)
        crown_x = (w - crown_w) // 2
        crown_h = int(h * 0.65)
        cv2.rectangle(hat, (crown_x, 0), (crown_x + crown_w, crown_h), 
                     (30, 30, 30, 255), -1)
        
        # Brim (wide part)
        brim_h = h - crown_h
        cv2.rectangle(hat, (0, crown_h), (w, h), (25, 25, 25, 255), -1)
        
        # Ribbon
        ribbon_y = int(crown_h * 0.85)
        ribbon_h = max(8, crown_h // 10)
        cv2.rectangle(hat, (crown_x, ribbon_y), (crown_x + crown_w, ribbon_y + ribbon_h),
                     (200, 50, 50, 255), -1)
        
        # Add shine gradient
        for i in range(crown_h):
            alpha = int(30 * (1 - i / crown_h))
            hat[i, crown_x:crown_x + crown_w // 3, :3] += alpha
        
        return hat
    
    def _generate_party_hat(self) -> ArrayU8:
        """Generate a colorful party hat (cone shape)."""
        w, h = 280, 250
        hat = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Create cone triangle
        points = np.array([
            [w // 2, 0],  # top
            [0, h],  # bottom left
            [w, h],  # bottom right
        ], dtype=np.int32)
        
        # Fill with gradient colors
        for y in range(h):
            ratio = y / h
            color = (
                int(255 * (1 - ratio) + 50 * ratio),  # Blue to purple
                int(100 * (1 - ratio) + 200 * ratio),  # Green
                int(200),  # Red constant
                255
            )
            left_x = int(w // 2 - (w // 2) * ratio)
            right_x = int(w // 2 + (w // 2) * ratio)
            hat[y, left_x:right_x] = color
        
        # Add stripes
        for i in range(0, h, 25):
            stripe_color = (255, 215, 0, 255) if i % 50 == 0 else (255, 255, 255, 255)
            for y in range(max(0, i - 3), min(h, i + 3)):
                ratio = y / h
                left_x = max(0, int(w // 2 - (w // 2) * ratio))
                right_x = min(w, int(w // 2 + (w // 2) * ratio))
                hat[y, left_x:right_x] = stripe_color
        
        # Add pom-pom at top
        cv2.circle(hat, (w // 2, 15), 15, (255, 100, 200, 255), -1)
        
        return hat
    
    def _generate_crown(self) -> ArrayU8:
        """Generate a royal crown."""
        w, h = 320, 180
        hat = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Base band
        band_h = int(h * 0.4)
        cv2.rectangle(hat, (0, h - band_h), (w, h), (255, 215, 0, 255), -1)
        
        # Points/spikes
        num_points = 5
        point_width = w // num_points
        for i in range(num_points):
            x_center = i * point_width + point_width // 2
            points = np.array([
                [x_center - point_width // 3, h - band_h],
                [x_center, 0],
                [x_center + point_width // 3, h - band_h],
            ], dtype=np.int32)
            cv2.fillPoly(hat, [points], (255, 215, 0, 255))
            
            # Jewels on each point
            jewel_colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 0, 200), (0, 200, 200)]
            cv2.circle(hat, (x_center, h // 3), 12, (*jewel_colors[i], 255), -1)
        
        # Add metallic shine
        for y in range(h - band_h, h):
            shine = int(60 * (1 - (y - (h - band_h)) / band_h))
            hat[y, :, :3] = np.clip(hat[y, :, :3].astype(int) + shine, 0, 255).astype(np.uint8)
        
        return hat
    
    def start_camera(self) -> bool:
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def detect_people_with_depth(self, frame_bgr: ArrayU8, timestamp_ms: int) -> List[Person3D]:
        """Detect people and estimate their real heights accounting for depth."""
        h, w = frame_bgr.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect poses
        result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        people = []
        for idx, world_landmarks in enumerate(result.pose_world_landmarks):
            if idx >= len(result.pose_landmarks):
                continue
            
            norm_landmarks = result.pose_landmarks[idx]
            
            # Convert normalized landmarks to pixel coordinates
            landmarks_2d = np.array([
                [lm.x * w, lm.y * h] for lm in norm_landmarks
            ], dtype=np.float32)
            
            # Extract 3D world coordinates (in meters from MediaPipe)
            landmarks_3d = np.array([
                [lm.x, lm.y, lm.z] for lm in world_landmarks
            ], dtype=np.float32)
            
            visibility = np.array([lm.visibility for lm in norm_landmarks], dtype=np.float32)
            
            # Calculate depth (average Z coordinate of visible landmarks)
            visible_mask = visibility > 0.5
            if not np.any(visible_mask):
                continue
            
            depth_z = np.mean(np.abs(landmarks_3d[visible_mask, 2]))
            
            # Calculate pixel height
            valid_y = landmarks_2d[visible_mask, 1]
            height_pixels = float(valid_y.max() - valid_y.min())
            
            # Estimate real height using depth
            # MediaPipe world landmarks are in meters, hip-to-head is roughly person's height
            # Use shoulder/hip landmarks to estimate height in world coordinates
            left_shoulder = 11
            right_shoulder = 12
            left_hip = 23
            right_hip = 24
            left_ankle = 27
            right_ankle = 28
            
            # Calculate world-space height (meters)
            height_landmarks = [left_shoulder, right_shoulder, left_hip, right_hip, 
                              left_ankle, right_ankle]
            valid_heights = [landmarks_3d[i, 1] for i in height_landmarks 
                           if i < len(visibility) and visibility[i] > 0.5]
            
            if valid_heights:
                world_height = max(valid_heights) - min(valid_heights)
                real_height_estimate = abs(world_height)  # meters
            else:
                real_height_estimate = height_pixels * (depth_z + 1.0) / 500.0
            
            # Get head information
            nose_idx = 0
            left_ear_idx = 7
            right_ear_idx = 8
            head_landmarks = [nose_idx, left_ear_idx, right_ear_idx]
            
            head_points = [landmarks_2d[i] for i in head_landmarks 
                          if i < len(visibility) and visibility[i] > 0.5]
            
            if head_points:
                head_points = np.array(head_points)
                head_top_y = int(head_points[:, 1].min())
                head_center_x = int(np.mean(head_points[:, 0]))
                head_width = float(head_points[:, 0].max() - head_points[:, 0].min())
            else:
                head_top_y = int(landmarks_2d[nose_idx, 1])
                head_center_x = int(landmarks_2d[nose_idx, 0])
                head_width = height_pixels * 0.15
            
            # Calculate shoulder width for better hat placement
            if visibility[left_shoulder] > 0.5 and visibility[right_shoulder] > 0.5:
                shoulder_width = abs(landmarks_2d[left_shoulder, 0] - 
                                   landmarks_2d[right_shoulder, 0])
            else:
                shoulder_width = head_width * 2.0
            
            person = Person3D(
                index=idx,
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d,
                visibility=visibility,
                height_pixels=height_pixels,
                depth_z=depth_z,
                real_height_estimate=real_height_estimate,
                head_top_y=head_top_y,
                head_center_x=head_center_x,
                head_width=head_width,
                shoulder_width=shoulder_width,
            )
            people.append(person)
        
        return people
    
    def find_shortest_person(self, people: List[Person3D]) -> Optional[int]:
        """Find the shortest person accounting for depth."""
        if len(people) < 2:
            return None
        
        # Use real height estimate (depth-corrected)
        shortest_idx = min(range(len(people)), 
                          key=lambda i: people[i].real_height_estimate)
        
        return shortest_idx
    
    def find_tallest_person(self, people: List[Person3D]) -> Optional[int]:
        """Find the tallest person accounting for depth."""
        if not people:
            return None
        
        tallest_idx = max(range(len(people)), 
                         key=lambda i: people[i].real_height_estimate)
        
        return tallest_idx
    
    def stretch_face(
        self, 
        frame: ArrayU8, 
        person: Person3D, 
        stretch_factor: float
    ) -> ArrayU8:
        """Stretch a person's face/upper body vertically."""
        nose_idx = 0
        left_shoulder = 11
        right_shoulder = 12
        
        # Define face region (nose to shoulders)
        face_points = []
        for idx in [nose_idx, left_shoulder, right_shoulder]:
            if person.visibility[idx] > 0.5:
                face_points.append(person.landmarks_2d[idx])
        
        if len(face_points) < 2:
            return frame
        
        face_points = np.array(face_points)
        
        # Create bounding box for face/upper body
        x_min = int(face_points[:, 0].min() - person.head_width)
        x_max = int(face_points[:, 0].max() + person.head_width)
        y_min = int(face_points[:, 1].min() - person.head_width)
        y_max = int(face_points[:, 1].max() + person.head_width * 2)
        
        # Clamp to frame bounds
        h, w = frame.shape[:2]
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return frame
        
        # Extract region
        region = frame[y_min:y_max, x_min:x_max].copy()
        
        # Calculate stretched height
        original_h = y_max - y_min
        stretched_h = int(original_h * stretch_factor)
        
        if stretched_h <= original_h:
            return frame
        
        # Resize region
        region_w = x_max - x_min
        stretched_region = cv2.resize(
            region, 
            (region_w, stretched_h), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Create output frame
        output = frame.copy()
        
        # Calculate placement (anchor at bottom of original region)
        new_y_min = y_max - stretched_h
        new_y_max = y_max
        
        # Clamp again
        if new_y_min < 0:
            trim = -new_y_min
            stretched_region = stretched_region[trim:, :]
            new_y_min = 0
        
        if new_y_max > h:
            trim = new_y_max - h
            stretched_region = stretched_region[:-trim, :]
            new_y_max = h
        
        # Blend with feathered edges for smooth transition
        blend_size = min(20, (new_y_max - new_y_min) // 4)
        
        # Top feather
        for i in range(blend_size):
            alpha = i / blend_size
            output[new_y_min + i, x_min:x_max] = (
                alpha * stretched_region[i] + 
                (1 - alpha) * output[new_y_min + i, x_min:x_max]
            ).astype(np.uint8)
        
        # Middle (full replacement)
        output[new_y_min + blend_size:new_y_max, x_min:x_max] = \
            stretched_region[blend_size:, :]
        
        return output
    
    def add_3d_hat(
        self, 
        frame: ArrayU8, 
        person: Person3D, 
        height_diff: float,
        hat_index: int = 0
    ) -> ArrayU8:
        """Add a 3D hat that compensates for height difference."""
        if height_diff < 1.0:
            return frame
        
        # Select hat
        hat = self.hat_models[hat_index % len(self.hat_models)].copy()
        
        # Scale hat based on person's size and depth
        # Closer people (smaller depth_z) need larger hats
        depth_scale = 1.0 / (person.depth_z + 0.5)
        base_width = person.shoulder_width * 0.8
        
        # Hat should be tall enough to compensate for height difference
        # but also proportional to head width
        target_height = max(height_diff * 0.8, base_width * 0.5)
        
        # Calculate scale factors
        scale_x = base_width / hat.shape[1]
        scale_y = target_height / hat.shape[0]
        scale = max(scale_x, scale_y, 0.3) * depth_scale
        
        # Resize hat
        new_w = max(1, int(hat.shape[1] * scale))
        new_h = max(1, int(hat.shape[0] * scale))
        resized_hat = cv2.resize(hat, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate position
        hat_h, hat_w = resized_hat.shape[:2]
        x = person.head_center_x - hat_w // 2
        y = person.head_top_y - hat_h
        
        # Overlay hat with alpha blending
        output = self._overlay_rgba(frame, resized_hat, (x, y))
        
        return output
    
    def _overlay_rgba(
        self, 
        frame: ArrayU8, 
        overlay: ArrayU8, 
        pos: Tuple[int, int]
    ) -> ArrayU8:
        """Overlay RGBA image onto BGR frame."""
        x, y = pos
        h, w = frame.shape[:2]
        oh, ow = overlay.shape[:2]
        
        # Calculate valid region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + ow)
        y2 = min(h, y + oh)
        
        if x1 >= x2 or y1 >= y2:
            return frame
        
        # Calculate overlay region
        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)
        
        # Extract regions
        output = frame.copy()
        roi = output[y1:y2, x1:x2]
        overlay_rgb = overlay[oy1:oy2, ox1:ox2, :3]
        alpha = overlay[oy1:oy2, ox1:ox2, 3:4] / 255.0
        
        # Blend
        blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        output[y1:y2, x1:x2] = blended
        
        return output
    
    def draw_debug_info(
        self, 
        frame: ArrayU8, 
        people: List[Person3D],
        shortest_idx: Optional[int],
        tallest_idx: Optional[int]
    ) -> ArrayU8:
        """Draw debug overlays showing detections and measurements."""
        output = frame.copy()
        
        for person in people:
            color = (0, 255, 0)  # Green default
            
            if person.index == shortest_idx:
                color = (0, 0, 255)  # Red for shortest
            elif person.index == tallest_idx:
                color = (255, 0, 0)  # Blue for tallest
            
            # Draw skeleton
            for i, (x, y) in enumerate(person.landmarks_2d):
                if person.visibility[i] > 0.5:
                    cv2.circle(output, (int(x), int(y)), 3, color, -1)
            
            # Draw info text
            info_x = int(person.landmarks_2d[0, 0])
            info_y = int(person.landmarks_2d[0, 1]) - 30
            
            cv2.putText(
                output,
                f"P{person.index} H:{person.real_height_estimate:.2f}m",
                (info_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            cv2.putText(
                output,
                f"D:{person.depth_z:.2f}m",
                (info_x, info_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        return output
    
    def process_frame(
        self, 
        frame: ArrayU8, 
        method: str = "hat",
        debug: bool = False
    ) -> ArrayU8:
        """Process a single frame."""
        # Detect people with depth
        timestamp_ms = int(self.total_frame_count * 33.33)  # Use total count for monotonic timestamps
        people = self.detect_people_with_depth(frame, timestamp_ms)
        
        if len(people) < 2:
            # Not enough people to equalize
            if debug and people:
                frame = self.draw_debug_info(frame, people, None, None)
            return frame
        
        # Find shortest and tallest
        shortest_idx = self.find_shortest_person(people)
        tallest_idx = self.find_tallest_person(people)
        
        if shortest_idx is None or tallest_idx is None:
            return frame
        
        shortest = people[shortest_idx]
        tallest = people[tallest_idx]
        
        # Calculate height difference in pixels
        # Need to adjust for depth - closer people appear larger
        height_diff_pixels = tallest.height_pixels - shortest.height_pixels
        
        # Adjust for depth difference
        depth_ratio = tallest.depth_z / (shortest.depth_z + 0.001)
        adjusted_diff = height_diff_pixels / depth_ratio
        
        # Apply equalization
        if method == "stretch":
            stretch_factor = 1.0 + (adjusted_diff / shortest.height_pixels) * 0.5
            frame = self.stretch_face(frame, shortest, stretch_factor)
        
        elif method == "hat":
            # Cycle through hat designs based on time
            hat_index = (self.frame_count // 30) % len(self.hat_models)
            frame = self.add_3d_hat(frame, shortest, abs(adjusted_diff), hat_index)
        
        elif method == "both":
            # Stretch a bit and add a smaller hat
            stretch_factor = 1.0 + (adjusted_diff / shortest.height_pixels) * 0.3
            frame = self.stretch_face(frame, shortest, stretch_factor)
            hat_index = (self.frame_count // 30) % len(self.hat_models)
            frame = self.add_3d_hat(frame, shortest, abs(adjusted_diff) * 0.5, hat_index)
        
        # Draw debug info if requested
        if debug:
            frame = self.draw_debug_info(frame, people, shortest_idx, tallest_idx)
        
        return frame
    
    def run(self, method: str = "hat", debug: bool = True):
        """Run the real-time video loop."""
        if not self.start_camera():
            print("Failed to open camera!")
            return
        
        print("Real-time Height Equalizer Started!")
        print(f"Mode: {method}")
        print("Controls:")
        print("  'h' - Hat mode")
        print("  's' - Stretch mode")
        print("  'b' - Both mode")
        print("  'd' - Toggle debug overlay")
        print("  'q' - Quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                self.total_frame_count += 1  # Never reset - for monotonic timestamps
                
                # Process frame
                processed = self.process_frame(frame, method, debug)
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Draw FPS
                cv2.putText(
                    processed,
                    f"FPS: {self.fps:.1f} | Mode: {method}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Display
                cv2.imshow("Height Equalizer - Justice for Short Kings!", processed)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    method = "hat"
                elif key == ord('s'):
                    method = "stretch"
                elif key == ord('b'):
                    method = "both"
                elif key == ord('d'):
                    debug = not debug
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-time height equalizer with depth awareness"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to MediaPipe pose_landmarker model (.task file)"
    )
    parser.add_argument(
        "--hat",
        type=Path,
        help="Optional custom hat model (transparent PNG)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--method",
        choices=["hat", "stretch", "both"],
        default="hat",
        help="Height equalization method"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug overlays"
    )
    
    args = parser.parse_args()
    
    equalizer = RealtimeHeightEqualizer(
        pose_model_path=args.model,
        hat_model_path=args.hat,
        camera_index=args.camera,
    )
    
    equalizer.run(method=args.method, debug=args.debug)


if __name__ == "__main__":
    main()