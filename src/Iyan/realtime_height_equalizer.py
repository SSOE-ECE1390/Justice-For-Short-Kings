from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
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


class HatStyle(str, Enum):
    TOP_HAT = "top_hat"
    PARTY_HAT = "party_hat"
    CROWN = "crown"


class RealtimeHeightEqualizer:
    """Real-time video processor that equalizes heights with depth awareness."""

    def __init__(
        self,
        pose_model_path: Path | str,
        *,
        hat_model_path: Optional[Path | str] = None,
        hat_style: str | HatStyle = "top_hat",
        camera_index: int = 0,
        max_people: int = 5,
        min_confidence: float = 0.4,
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
        )

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.camera_index = camera_index
        self.cap = None

        # Load or generate hat models (top, party, crown)
        self.hat_models = self._load_hat_models(hat_model_path)

        if isinstance(hat_style, HatStyle):
            self.hat_style = hat_style
        else:
            try:
                self.hat_style = HatStyle(hat_style)
            except ValueError:
                self.hat_style = HatStyle.TOP_HAT

        if len(self.hat_models) >= 3:
            style_to_index = {
                HatStyle.TOP_HAT: 0,
                HatStyle.PARTY_HAT: 1,
                HatStyle.CROWN: 2,
            }
            self.current_hat_index = style_to_index.get(self.hat_style, 0)
        else:
            self.current_hat_index = 0

        self.frame_count = 0
        self.total_frame_count = 0
        self.fps = 0
        self.last_time = time.time()

    # --------------------------------------------------------------------- IO

    def _load_hat_models(self, hat_path: Optional[Path | str]) -> List[ArrayU8]:
        hats: List[ArrayU8] = []

        if hat_path and Path(hat_path).exists():
            hat = cv2.imread(str(hat_path), cv2.IMREAD_UNCHANGED)
            if hat is not None and hat.shape[2] == 4:
                hats.append(hat)

        if not hats:
            hats.extend(
                [
                    self._generate_top_hat(),
                    self._generate_party_hat(),
                    self._generate_crown(),
                ]
            )

        return hats

    # ---------------------------------------------------------------- hat gens

    def _generate_top_hat(self) -> ArrayU8:
        w, h = 300, 200
        hat = np.zeros((h, w, 4), dtype=np.uint8)

        crown_w = int(w * 0.5)
        crown_x = (w - crown_w) // 2
        crown_h = int(h * 0.65)
        cv2.rectangle(
            hat,
            (crown_x, 0),
            (crown_x + crown_w, crown_h),
            (30, 30, 30, 255),
            -1,
        )

        cv2.rectangle(
            hat,
            (0, crown_h),
            (w, h),
            (25, 25, 25, 255),
            -1,
        )

        ribbon_y = int(crown_h * 0.85)
        ribbon_h = max(8, crown_h // 10)
        cv2.rectangle(
            hat,
            (crown_x, ribbon_y),
            (crown_x + crown_w, ribbon_y + ribbon_h),
            (200, 50, 50, 255),
            -1,
        )

        for i in range(crown_h):
            alpha = int(30 * (1 - i / crown_h))
            hat[i, crown_x : crown_x + crown_w // 3, :3] = np.clip(
                hat[i, crown_x : crown_x + crown_w // 3, :3].astype(int) + alpha,
                0,
                255,
            )

        return hat

    def _generate_party_hat(self) -> ArrayU8:
        w, h = 220, 260
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.uint8)

        base_y = h - 40
        apex_y = 10
        pts = np.array(
            [
                [w // 2, apex_y],
                [20, base_y],
                [w - 20, base_y],
            ],
            np.int32,
        )

        cv2.fillConvexPoly(alpha, pts, 255)

        stripe_colors = [
            (255, 200, 200),
            (230, 170, 230),
            (200, 180, 255),
            (200, 220, 255),
            (200, 255, 220),
            (240, 240, 180),
            (250, 220, 140),
        ]
        stripe_height = max(3, (base_y - apex_y) // len(stripe_colors))

        stripes_rgb = np.zeros_like(rgb)
        for i, color in enumerate(stripe_colors):
            y1 = base_y - (i + 1) * stripe_height
            y0 = base_y - i * stripe_height
            y1 = max(apex_y, y1)
            y0 = max(apex_y, y0)
            cv2.rectangle(
                stripes_rgb,
                (0, y1),
                (w - 1, y0),
                color,
                thickness=-1,
            )

        mask_bool = alpha > 0
        rgb[mask_bool] = stripes_rgb[mask_bool]

        pom_color = (210, 120, 255)
        pom_center = (w // 2, apex_y)
        pom_radius = 10
        cv2.circle(alpha, pom_center, pom_radius, 255, -1)
        cv2.circle(rgb, pom_center, pom_radius, pom_color, -1)

        hat = np.dstack([rgb, alpha])
        return hat

    def _generate_crown(self) -> ArrayU8:
        w, h = 260, 160
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.uint8)

        base_top = int(h * 0.55)
        base_color = (0, 240, 255)

        rgb[base_top:h, :, :] = base_color
        cv2.rectangle(alpha, (0, base_top), (w - 1, h - 1), 255, -1)

        spike_count = 5
        spacing = w // (spike_count + 1)
        spike_width = int(spacing * 0.8)
        spike_height = int(h * 0.55)

        for i in range(spike_count):
            center_x = (i + 1) * spacing
            left = center_x - spike_width // 2
            right = center_x + spike_width // 2
            top = base_top - spike_height

            pts = np.array(
                [
                    [center_x, top],
                    [left, base_top],
                    [right, base_top],
                ],
                np.int32,
            )

            cv2.fillConvexPoly(alpha, pts, 255)
            cv2.fillConvexPoly(rgb, pts, base_color)

        jewel_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 255, 0),
        ]
        for i in range(spike_count):
            center_x = (i + 1) * spacing
            jewel_y = base_top - spike_height // 2
            color = jewel_colors[i % len(jewel_colors)]

            cv2.circle(alpha, (center_x, jewel_y), 7, 255, -1)
            cv2.circle(rgb, (center_x, jewel_y), 7, color, -1)

        hat = np.dstack([rgb, alpha])
        return hat

    # ---------------------------------------------------------------- camera

    def start_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        return True

    # ----------------------------------------------------------- pose + depth

    def detect_people_with_depth(
        self, frame_bgr: ArrayU8, timestamp_ms: int
    ) -> List[Person3D]:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        people: List[Person3D] = []
        for idx, world_landmarks in enumerate(result.pose_world_landmarks):
            if idx >= len(result.pose_landmarks):
                continue

            norm_landmarks = result.pose_landmarks[idx]

            landmarks_2d = np.array(
                [[lm.x * w, lm.y * h] for lm in norm_landmarks], dtype=np.float32
            )
            landmarks_3d = np.array(
                [[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32
            )
            visibility = np.array(
                [getattr(lm, "visibility", 1.0) for lm in norm_landmarks],
                dtype=np.float32,
            )

            visible_mask = visibility > 0.5
            if not np.any(visible_mask):
                continue

            visible_points = landmarks_2d[visible_mask]
            visible_y = visible_points[:, 1]
            height_pixels = float(visible_y.max() - visible_y.min())

            left_shoulder_idx = 11
            right_shoulder_idx = 12

            if (
                visibility[left_shoulder_idx] > 0.5
                and visibility[right_shoulder_idx] > 0.5
            ):
                left_shoulder = landmarks_2d[left_shoulder_idx]
                right_shoulder = landmarks_2d[right_shoulder_idx]
                shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
                depth_z = float(np.mean(landmarks_3d[visible_mask, 2]))
            else:
                shoulder_width = height_pixels * 0.25
                depth_z = 2.0

            depth_scale = shoulder_width / 100.0
            real_height = height_pixels / max(depth_scale, 0.1)

            nose_idx = 0
            left_eye_idx = 2
            right_eye_idx = 5

            head_y_coords = []
            head_x_coords = []
            for head_idx in [nose_idx, left_eye_idx, right_eye_idx]:
                if visibility[head_idx] > 0.5:
                    head_y_coords.append(landmarks_2d[head_idx, 1])
                    head_x_coords.append(landmarks_2d[head_idx, 0])

            if head_y_coords:
                head_top_y = int(min(head_y_coords))
                head_center_x = int(np.mean(head_x_coords))
                head_width = shoulder_width * 0.6
            else:
                head_top_y = int(visible_y.min())
                head_center_x = int(np.mean(landmarks_2d[visible_mask, 0]))
                head_width = height_pixels * 0.15

            people.append(
                Person3D(
                    index=idx,
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    visibility=visibility,
                    height_pixels=height_pixels,
                    depth_z=depth_z,
                    real_height_estimate=real_height,
                    head_top_y=head_top_y,
                    head_center_x=head_center_x,
                    head_width=head_width,
                    shoulder_width=shoulder_width,
                )
            )

        return people

    # ---------------------------------------------------------- helpers

    def find_shortest_person(self, people: List[Person3D]) -> Optional[int]:
        if not people:
            return None
        min_height = float("inf")
        shortest_idx = None
        for person in people:
            if person.real_height_estimate < min_height:
                min_height = person.real_height_estimate
                shortest_idx = person.index
        return shortest_idx

    def find_tallest_person(self, people: List[Person3D]) -> Optional[int]:
        if not people:
            return None
        max_height = -float("inf")
        tallest_idx = None
        for person in people:
            if person.real_height_estimate > max_height:
                max_height = person.real_height_estimate
                tallest_idx = person.index
        return tallest_idx

    # ------------------------------------------------------------- stretch

    def stretch_person(
        self,
        frame: ArrayU8,
        person: Person3D,
        stretch_factor: float,
    ) -> ArrayU8:
        """Stretch a person vertically, anchoring the feet and avoiding head cut-off."""
        if stretch_factor <= 1.0:
            return frame

        h, w = frame.shape[:2]
        output = frame.copy()

        # Person bbox from visible landmarks
        visible_mask = person.visibility > 0.5
        if not np.any(visible_mask):
            return frame

        visible_points = person.landmarks_2d[visible_mask]
        x_min = int(max(0, visible_points[:, 0].min() - 20))
        x_max = int(min(w, visible_points[:, 0].max() + 20))
        y_min = int(max(0, visible_points[:, 1].min() - 20))
        y_max = int(min(h, visible_points[:, 1].max() + 20))

        if x_max <= x_min or y_max <= y_min:
            return frame

        roi = frame[y_min:y_max, x_min:x_max]
        roi_h, roi_w = roi.shape[:2]
        if roi_h <= 0 or roi_w <= 0:
            return frame

        # ---- NEW: clamp stretch so full stretched person fits on screen ----
        bottom_y = y_max  # feet anchor
        # maximum vertical factor so new_height <= bottom_y
        max_factor_geom = (bottom_y - 1) / roi_h
        max_factor_geom = max(1.0, max_factor_geom)
        stretch_factor = float(min(stretch_factor, max_factor_geom))
        # --------------------------------------------------------------------

        new_height = int(roi_h * stretch_factor)
        new_height = max(roi_h + 1, min(new_height, h))

        stretched = cv2.resize(
            roi,
            (roi_w, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        bottom_y = y_max
        dest_top = bottom_y - new_height
        dest_bottom = bottom_y
        if dest_top < 0:
            dest_top = 0
            dest_bottom = new_height
        dest_h = dest_bottom - dest_top
        if dest_h <= 0:
            return frame

        src = stretched[-dest_h:, :, :]

        edge_width = min(10, roi_w // 4)
        alpha = np.ones((dest_h, roi_w), dtype=np.float32)
        if edge_width > 0:
            for i in range(edge_width):
                t = i / edge_width
                alpha[:, i] *= t
                alpha[:, -(i + 1)] *= t
        alpha = alpha[..., None]

        dest_patch = output[dest_top:dest_bottom, x_min:x_max]
        blended = (src * alpha + dest_patch * (1.0 - alpha)).astype(np.uint8)

        output[dest_top:dest_bottom, x_min:x_max] = blended
        return output

    # -------------------------------------------------------------- hats

    def add_3d_hat(
        self,
        frame: ArrayU8,
        person: Person3D,
        height_gap: float,
        hat_index: Optional[int] = None,
    ) -> ArrayU8:
        if not self.hat_models:
            return frame

        if hat_index is None:
            hat_index = self.current_hat_index

        hat = self.hat_models[hat_index % len(self.hat_models)].copy()
        output = frame.copy()

        base_h, base_w = hat.shape[:2]

        height_gap = max(0.0, float(height_gap))
        scale_for_width = (person.head_width * 1.4) / max(base_w, 1)
        scale_for_gap = (0.7 * height_gap) / max(base_h, 1) if height_gap > 0 else 0.0

        scale = max(scale_for_width, scale_for_gap, 0.4)

        depth_scale = 1.5 / max(person.depth_z, 0.6)
        depth_scale = float(np.clip(depth_scale, 0.7, 1.4))
        scale *= depth_scale
        scale = float(np.clip(scale, 0.5, 1.6))

        new_w = max(10, int(base_w * scale))
        new_h = max(10, int(base_h * scale))

        resized_hat = cv2.resize(hat, (new_w, new_h), interpolation=cv2.INTER_AREA)

        hat_h, hat_w = resized_hat.shape[:2]
        x_pos = int(person.head_center_x - hat_w // 2)

        vertical_offset = int(0.2 * hat_h)  # push hat up a bit
        y_pos = int(person.head_top_y - hat_h - vertical_offset)

        output = self._overlay_rgba(output, resized_hat, (x_pos, y_pos))
        return output

    def _overlay_rgba(
        self,
        base: ArrayU8,
        overlay: ArrayU8,
        top_left: Tuple[int, int],
    ) -> ArrayU8:
        x, y = top_left
        h, w = overlay.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(base.shape[1], x + w)
        y2 = min(base.shape[0], y + h)

        if x1 >= x2 or y1 >= y2:
            return base

        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        overlay_rgb = overlay[oy1:oy2, ox1:ox2, :3]
        alpha = overlay[oy1:oy2, ox1:ox2, 3:4] / 255.0

        roi = base[y1:y2, x1:x2]

        blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        output = base.copy()
        output[y1:y2, x1:x2] = blended

        return output

    # ---------------------------------------------------------- debug drawing

    def draw_debug_info(
        self,
        frame: ArrayU8,
        people: List[Person3D],
        shortest_idx: Optional[int],
        tallest_idx: Optional[int],
    ) -> ArrayU8:
        output = frame.copy()

        for person in people:
            color = (0, 255, 0)
            if person.index == shortest_idx:
                color = (0, 0, 255)
            elif person.index == tallest_idx:
                color = (255, 0, 0)

            for i, (x, y) in enumerate(person.landmarks_2d):
                if person.visibility[i] > 0.5:
                    cv2.circle(output, (int(x), int(y)), 3, color, -1)

            info_x = int(person.landmarks_2d[0, 0])
            info_y = int(person.landmarks_2d[0, 1]) - 30

            cv2.putText(
                output,
                f"P{person.index} H:{person.real_height_estimate:.2f}",
                (info_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            cv2.putText(
                output,
                f"D:{person.depth_z:.2f}m",
                (info_x, info_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return output

    # ------------------------------------------------------------- frame proc

    def process_frame(
        self,
        frame: ArrayU8,
        method: str = "hat",
        debug: bool = False,
    ) -> ArrayU8:
        timestamp_ms = int(self.total_frame_count * 33.33)
        people = self.detect_people_with_depth(frame, timestamp_ms)

        if len(people) < 2:
            if debug and people:
                frame = self.draw_debug_info(frame, people, None, None)
            return frame

        shortest_idx = self.find_shortest_person(people)
        tallest_idx = self.find_tallest_person(people)

        if (
            shortest_idx is None
            or tallest_idx is None
            or shortest_idx == tallest_idx
        ):
            return frame

        idx_to_person = {p.index: p for p in people}
        shortest = idx_to_person[shortest_idx]
        tallest = idx_to_person[tallest_idx]

        height_diff_pixels = max(
            0.0, float(tallest.height_pixels - shortest.height_pixels)
        )

        if method == "stretch":
            if shortest.height_pixels > 1.0:
                base_factor = tallest.height_pixels / shortest.height_pixels
            else:
                base_factor = 1.0
            stretch_factor = float(np.clip(base_factor, 1.0, 1.6))
            frame = self.stretch_person(frame, shortest, stretch_factor)

        elif method == "hat":
            frame = self.add_3d_hat(frame, shortest, abs(height_diff_pixels))

        elif method == "both":
            if shortest.height_pixels > 1.0:
                base_factor = tallest.height_pixels / shortest.height_pixels
            else:
                base_factor = 1.0
            stretch_factor = 1.0 + (base_factor - 1.0) * 0.5
            stretch_factor = float(np.clip(stretch_factor, 1.0, 1.4))
            frame = self.stretch_person(frame, shortest, stretch_factor)
            frame = self.add_3d_hat(frame, shortest, abs(height_diff_pixels) * 0.5)

        if debug:
            frame = self.draw_debug_info(frame, people, shortest_idx, tallest_idx)

        return frame

    # ------------------------------------------------------------------- main

    def run(self, method: str = "hat", debug: bool = True):
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
        print("  '1' - Use TOP HAT")
        print("  '2' - Use PARTY HAT")
        print("  '3' - Use CROWN")
        print("  'q' - Quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                self.total_frame_count += 1

                processed = self.process_frame(frame, method, debug)

                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time

                if len(self.hat_models) >= 3:
                    idx = self.current_hat_index % 3
                    hat_label = ["TOP", "PARTY", "CROWN"][idx]
                else:
                    hat_label = "CUSTOM"

                cv2.putText(
                    processed,
                    f"FPS: {self.fps:.1f} | Mode: {method} | Hat: {hat_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Height Equalizer - Justice for Short Kings!", processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("h"):
                    method = "hat"
                    print(f"Switched to: {method}")
                elif key == ord("s"):
                    method = "stretch"
                    print(f"Switched to: {method}")
                elif key == ord("b"):
                    method = "both"
                    print(f"Switched to: {method}")
                elif key == ord("d"):
                    debug = not debug
                    print(f"Debug: {'ON' if debug else 'OFF'}")
                elif key == ord("1"):
                    self.current_hat_index = 0
                    self.hat_style = HatStyle.TOP_HAT
                    print("Hat: TOP_HAT")
                elif key == ord("2"):
                    if len(self.hat_models) >= 2:
                        self.current_hat_index = 1
                        self.hat_style = HatStyle.PARTY_HAT
                        print("Hat: PARTY_HAT")
                elif key == ord("3"):
                    if len(self.hat_models) >= 3:
                        self.current_hat_index = 2
                        self.hat_style = HatStyle.CROWN
                        print("Hat: CROWN")

        finally:
            self.cleanup()

    def cleanup(self):
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
        help="Path to MediaPipe pose_landmarker model (.task file)",
    )
    parser.add_argument(
        "--hat",
        type=Path,
        help="Optional custom hat model (transparent PNG)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--method",
        choices=["hat", "stretch", "both"],
        default="hat",
        help="Height equalization method",
    )
    parser.add_argument(
        "--hat-style",
        choices=[s.value for s in HatStyle],
        default=HatStyle.TOP_HAT.value,
        help="Built-in hat style",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug overlays",
    )

    args = parser.parse_args()

    equalizer = RealtimeHeightEqualizer(
        pose_model_path=args.model,
        hat_model_path=args.hat,
        hat_style=args.hat_style,
        camera_index=args.camera,
    )

    equalizer.run(method=args.method, debug=args.debug)


if __name__ == "__main__":
    main()
