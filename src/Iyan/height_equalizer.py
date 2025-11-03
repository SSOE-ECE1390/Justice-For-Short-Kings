from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark


try:
    from mediapipe.solutions import pose as mp_pose  # type: ignore[attr-defined]
except ModuleNotFoundError:
    mp_pose = None


if mp_pose is not None:
    PoseLandmark = mp_pose.PoseLandmark
else:
    class PoseLandmark(IntEnum):
        NOSE = 0
        LEFT_EYE_INNER = 1
        LEFT_EYE = 2
        LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4
        RIGHT_EYE = 5
        RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7
        RIGHT_EAR = 8
        LEFT_MOUTH = 9
        RIGHT_MOUTH = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_PINKY = 17
        RIGHT_PINKY = 18
        LEFT_INDEX = 19
        RIGHT_INDEX = 20
        LEFT_THUMB = 21
        RIGHT_THUMB = 22
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32

NormalizedLandmark = mp_landmark.NormalizedLandmark
ArrayF32 = npt.NDArray[np.float32]
ArrayU8 = npt.NDArray[np.uint8]
ArrayI32 = npt.NDArray[np.int32]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned person bounding box represented with image coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    def clamp(self, frame_size: Tuple[int, int]) -> "BoundingBox":
        height, width = frame_size
        left = max(0, min(self.left, width - 1))
        right = max(left + 1, min(self.right, width))
        top = max(0, min(self.top, height - 1))
        bottom = max(top + 1, min(self.bottom, height))
        return BoundingBox(left, top, right, bottom)

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass
class PersonMeasurement:
    """Per-person pose metadata captured from MediaPipe."""

    index: int
    normalized_landmarks: Sequence[NormalizedLandmark]
    pixel_landmarks: ArrayF32  # shape (33, 2)
    visibility: ArrayF32  # shape (33,)
    frame_size: Tuple[int, int]
    segmentation_mask: Optional[ArrayF32] = None  # float mask [0,1]

    def _select_coords(self, min_visibility: float) -> ArrayF32:
        mask = self.visibility >= min_visibility
        if not np.any(mask):
            # Fallback to every landmark if nothing satisfied the threshold.
            mask = np.ones_like(self.visibility, dtype=bool)
        return self.pixel_landmarks[mask]

    def bounding_box(self, min_visibility: float = 0.2) -> BoundingBox:
        coords = self._select_coords(min_visibility)
        xs = coords[:, 0]
        ys = coords[:, 1]
        left = int(math.floor(xs.min()))
        right = int(math.ceil(xs.max()))
        top = int(math.floor(ys.min()))
        bottom = int(math.ceil(ys.max()))
        return BoundingBox(left, top, right, bottom).clamp(self.frame_size)

    @property
    def height_px(self) -> float:
        coords = self._select_coords(0.2)
        return float(coords[:, 1].max() - coords[:, 1].min())

    def head_top(self) -> Optional[int]:
        frame_h, _ = self.frame_size
        head_indices = [
            PoseLandmark.NOSE.value,
            PoseLandmark.LEFT_EYE_OUTER.value,
            PoseLandmark.RIGHT_EYE_OUTER.value,
            PoseLandmark.LEFT_EAR.value,
            PoseLandmark.RIGHT_EAR.value,
        ]
        valid_points = []
        for idx in head_indices:
            if idx >= len(self.visibility):
                continue
            if self.visibility[idx] < 0.2:
                continue
            valid_points.append(self.pixel_landmarks[idx, 1])
        if not valid_points:
            return None
        y_top = max(0, min(valid_points))
        return int(min(y_top, frame_h - 1))

    def head_width(self) -> Optional[float]:
        pairs = [
            (PoseLandmark.LEFT_EAR.value, PoseLandmark.RIGHT_EAR.value),
            (PoseLandmark.LEFT_EYE_OUTER.value, PoseLandmark.RIGHT_EYE_OUTER.value),
            (PoseLandmark.LEFT_EYE.value, PoseLandmark.RIGHT_EYE.value),
        ]
        distances = []
        for left_idx, right_idx in pairs:
            if max(left_idx, right_idx) >= len(self.visibility):
                continue
            if self.visibility[left_idx] < 0.2 or self.visibility[right_idx] < 0.2:
                continue
            left_pt = self.pixel_landmarks[left_idx]
            right_pt = self.pixel_landmarks[right_idx]
            distances.append(abs(left_pt[0] - right_pt[0]))
        if not distances:
            return None
        return float(np.median(distances))


@dataclass
class HeightAdjustmentResult:
    output_image: ArrayU8
    measurements: Sequence[PersonMeasurement]
    shortest_index: int
    reference_index: int
    method: str
    applied: bool


class HeightEqualizer:
    """Detects people in an image and equalizes the shortest person's height."""

    def __init__(
        self,
        pose_model_path: Path | str,
        *,
        accessory_path: Optional[Path | str] = None,
        max_people: int = 6,
        min_confidence: float = 0.5,
        segmentation_threshold: float = 0.5,
    ) -> None:
        pose_model_path = Path(pose_model_path)
        if not pose_model_path.exists():
            raise FileNotFoundError(
                f"Pose landmarker model file was not found: {pose_model_path}"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=max_people,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=True,
        )

        self._pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self._segmentation_threshold = segmentation_threshold
        self._accessory_rgba = self._load_accessory(accessory_path)

    def close(self) -> None:
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def evaluate_heights(self, image_bgr: ArrayU8) -> List[PersonMeasurement]:
        """Detect people in the provided image and measure their pixel heights."""
        if image_bgr.ndim != 3:
            raise ValueError("Expected a BGR image with three channels.")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._pose_landmarker.detect(mp_image)
        pose_landmarks = result.pose_landmarks or []

        segmentation_masks: List[ArrayF32] = []
        if result.segmentation_masks:
            for mask in result.segmentation_masks:
                mask_array = np.array(mask.numpy_view(), dtype=np.float32)
                segmentation_masks.append(mask_array)

        frame_h, frame_w = image_bgr.shape[:2]
        measurements: List[PersonMeasurement] = []
        for idx, normalized in enumerate(pose_landmarks):
            pixel_landmarks = np.array(
                [[lm.x * frame_w, lm.y * frame_h] for lm in normalized],
                dtype=np.float32,
            )
            visibility = np.array(
                [getattr(lm, "visibility", 1.0) for lm in normalized],
                dtype=np.float32,
            )

            segmentation_mask = None
            if idx < len(segmentation_masks):
                segmentation_mask = segmentation_masks[idx]

            measurements.append(
                PersonMeasurement(
                    index=idx,
                    normalized_landmarks=normalized,
                    pixel_landmarks=pixel_landmarks,
                    visibility=visibility,
                    frame_size=(frame_h, frame_w),
                    segmentation_mask=segmentation_mask,
                )
            )

        return measurements

    def equalize(
        self,
        image_bgr: ArrayU8,
        *,
        method: str = "stretch",
        reference_index: Optional[int] = None,
        shortest_index: Optional[int] = None,
    ) -> HeightAdjustmentResult:
        """Equalize the shortest person's height to match a reference individual."""
        if image_bgr.ndim != 3:
            raise ValueError("Expected a BGR image with three channels.")

        working = image_bgr.copy()
        measurements = self.evaluate_heights(working)
        if not measurements:
            raise RuntimeError("No people detected in the provided image.")

        heights = [m.height_px for m in measurements]
        if shortest_index is None:
            shortest_index = int(np.argmin(heights))
        if reference_index is None:
            reference_index = int(np.argmax(heights))

        if shortest_index == reference_index:
            raise ValueError(
                "Reference person and shortest person refer to the same individual."
            )

        shortest_person = measurements[shortest_index]
        reference_person = measurements[reference_index]

        target_height = reference_person.height_px
        current_height = shortest_person.height_px
        if current_height <= 0:
            raise RuntimeError("Unable to determine a valid height for the shortest person.")

        if target_height <= current_height + 1.0:
            return HeightAdjustmentResult(
                output_image=working,
                measurements=measurements,
                shortest_index=shortest_index,
                reference_index=reference_index,
                method=method,
                applied=False,
            )

        method = method.lower()
        if method == "stretch":
            updated = self._stretch_person_to_height(
                working, shortest_person, target_height
            )
        elif method == "accessory":
            updated = self._add_accessory_to_match_height(
                working, shortest_person, target_height
            )
        else:
            raise ValueError(f"Unsupported equalization method: {method}")

        return HeightAdjustmentResult(
            output_image=updated,
            measurements=measurements,
            shortest_index=shortest_index,
            reference_index=reference_index,
            method=method,
            applied=True,
        )

    def _stretch_person_to_height(
        self,
        image_bgr: ArrayU8,
        person: PersonMeasurement,
        target_height: float,
    ) -> ArrayU8:
        bbox = person.bounding_box()
        bbox_height = bbox.height
        bbox_width = bbox.width
        if bbox_height <= 1 or bbox_width <= 1:
            return image_bgr

        desired_height = int(round(target_height))
        if desired_height <= bbox_height:
            return image_bgr

        region = image_bgr[bbox.top : bbox.bottom, bbox.left : bbox.right]
        mask = self._extract_mask(person, bbox)
        # Resize only the person (mask) within the bounding box.
        stretched_region = cv2.resize(
            region, (bbox_width, desired_height), interpolation=cv2.INTER_CUBIC
        )
        stretched_mask = cv2.resize(
            mask, (bbox_width, desired_height), interpolation=cv2.INTER_LINEAR
        )
        stretched_mask = np.clip(stretched_mask, 0.0, 1.0)

        base = image_bgr.copy()
        dest_bottom = bbox.bottom
        dest_top = dest_bottom - stretched_region.shape[0]
        src = stretched_region
        src_mask = stretched_mask

        if dest_top < 0:
            trim = -dest_top
            src = src[trim:, :, :]
            src_mask = src_mask[trim:, :]
            dest_top = 0

        if dest_bottom > base.shape[0]:
            trim = dest_bottom - base.shape[0]
            src = src[: src.shape[0] - trim, :, :]
            src_mask = src_mask[: src_mask.shape[0] - trim, :]
            dest_bottom = base.shape[0]

        if src.size == 0:
            return image_bgr

        dest_slice = base[dest_top:dest_bottom, bbox.left:bbox.right]
        src_mask_expanded = src_mask[..., None]
        blended = (
            src * src_mask_expanded + dest_slice * (1.0 - src_mask_expanded)
        )
        base[dest_top:dest_bottom, bbox.left:bbox.right] = blended.astype(np.uint8)
        return base

    def _add_accessory_to_match_height(
        self,
        image_bgr: ArrayU8,
        person: PersonMeasurement,
        target_height: float,
    ) -> ArrayU8:
        if self._accessory_rgba is None:
            self._accessory_rgba = self._generate_default_hat()

        gap = max(0.0, target_height - person.height_px)
        if gap < 1.0:
            return image_bgr

        hat = self._accessory_rgba.copy()
        head_width = person.head_width()
        if head_width is None:
            bbox = person.bounding_box()
            head_width = bbox.width * 0.6

        head_top = person.head_top()
        if head_top is None:
            bbox = person.bounding_box()
            head_top = bbox.top

        scale_vertical = gap / hat.shape[0]
        desired_width = head_width * 1.2
        scale_horizontal = desired_width / hat.shape[1]
        scale = max(scale_vertical, scale_horizontal, 0.5)
        target_size = (
            max(1, int(round(hat.shape[1] * scale))),
            max(1, int(round(hat.shape[0] * scale))),
        )
        resized_hat = cv2.resize(hat, target_size, interpolation=cv2.INTER_AREA)

        overlay = image_bgr.copy()
        hat_height = resized_hat.shape[0]
        hat_width = resized_hat.shape[1]
        center_x = int(round(person.pixel_landmarks[PoseLandmark.NOSE.value, 0]))
        top_y = max(0, head_top - hat_height)
        left_x = int(round(center_x - hat_width / 2))

        overlay = self._overlay_rgba(
            overlay, resized_hat, top_left=(left_x, top_y)
        )
        return overlay

    def _extract_mask(self, person: PersonMeasurement, bbox: BoundingBox) -> ArrayF32:
        if person.segmentation_mask is not None:
            mask = np.clip(person.segmentation_mask, 0.0, 1.0)
        else:
            mask = np.zeros(person.frame_size, dtype=np.float32)
            polygon = self._pose_polygon(person)
            if polygon.size > 0:
                cv2.fillConvexPoly(mask, polygon, 1.0)

        cropped = mask[bbox.top : bbox.bottom, bbox.left : bbox.right]
        if cropped.size == 0:
            return np.ones((bbox.height, bbox.width), dtype=np.float32)
        return cropped

    def _pose_polygon(self, person: PersonMeasurement) -> ArrayI32:
        landmark_ids = [
            PoseLandmark.LEFT_SHOULDER.value,
            PoseLandmark.RIGHT_SHOULDER.value,
            PoseLandmark.RIGHT_ELBOW.value,
            PoseLandmark.RIGHT_WRIST.value,
            PoseLandmark.RIGHT_HIP.value,
            PoseLandmark.RIGHT_KNEE.value,
            PoseLandmark.RIGHT_ANKLE.value,
            PoseLandmark.RIGHT_FOOT_INDEX.value,
            PoseLandmark.LEFT_FOOT_INDEX.value,
            PoseLandmark.LEFT_ANKLE.value,
            PoseLandmark.LEFT_KNEE.value,
            PoseLandmark.LEFT_HIP.value,
            PoseLandmark.LEFT_WRIST.value,
            PoseLandmark.LEFT_ELBOW.value,
        ]
        points = []
        for idx in landmark_ids:
            if idx >= len(person.visibility):
                continue
            if person.visibility[idx] < 0.2:
                continue
            coords = person.pixel_landmarks[idx]
            points.append([int(round(coords[0])), int(round(coords[1]))])
        if not points:
            return np.empty((0, 2), dtype=np.int32)
        hull = cv2.convexHull(np.array(points, dtype=np.int32))
        return hull.reshape(-1, 2)

    def _overlay_rgba(
        self, base_bgr: ArrayU8, overlay_rgba: ArrayU8, top_left: Tuple[int, int]
    ) -> ArrayU8:
        result = base_bgr.copy()
        overlay_rgb = overlay_rgba[..., :3]
        alpha_mask = overlay_rgba[..., 3] / 255.0

        x, y = top_left
        h, w = overlay_rgb.shape[:2]

        if x >= result.shape[1] or y >= result.shape[0]:
            return result

        x_end = min(result.shape[1], x + w)
        y_end = min(result.shape[0], y + h)
        x_start = max(0, x)
        y_start = max(0, y)

        overlay_x_start = x_start - x
        overlay_y_start = y_start - y
        overlay_x_end = overlay_x_start + (x_end - x_start)
        overlay_y_end = overlay_y_start + (y_end - y_start)

        if overlay_x_start >= overlay_x_end or overlay_y_start >= overlay_y_end:
            return result

        roi = result[y_start:y_end, x_start:x_end]
        overlay_part = overlay_rgb[
            overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end
        ]
        alpha_part = alpha_mask[
            overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end
        ][..., None]

        blended = overlay_part * alpha_part + roi * (1.0 - alpha_part)
        result[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
        return result

    def _generate_default_hat(self) -> ArrayU8:
        width, height = 220, 140
        hat = np.zeros((height, width, 4), dtype=np.uint8)
        # Create a simple top-hat accessory.
        brim_height = int(height * 0.3)
        crown_height = height - brim_height
        brim_color = (30, 30, 30, 255)
        crown_color = (45, 45, 45, 255)
        ribbon_color = (200, 40, 40, 255)

        hat[:crown_height, width // 4 : width * 3 // 4] = crown_color
        hat[crown_height:, :] = brim_color
        ribbon_top = crown_height // 2
        ribbon_bottom = ribbon_top + max(4, crown_height // 6)
        hat[ribbon_top:ribbon_bottom, width // 4 : width * 3 // 4] = ribbon_color

        alpha = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(alpha, (width // 4, 0), (width * 3 // 4, crown_height), 220, -1)
        cv2.rectangle(alpha, (0, crown_height), (width, height), 200, -1)
        hat[..., 3] = alpha
        return hat

    def _load_accessory(self, accessory_path: Optional[Path | str]) -> Optional[ArrayU8]:
        if accessory_path is None:
            return None
        accessory_path = Path(accessory_path)
        if not accessory_path.exists():
            raise FileNotFoundError(f"Accessory image not found: {accessory_path}")
        accessory = cv2.imread(str(accessory_path), cv2.IMREAD_UNCHANGED)
        if accessory is None:
            raise RuntimeError(f"Failed to load accessory image: {accessory_path}")
        if accessory.shape[2] == 3:
            alpha = np.full(accessory.shape[:2], 255, dtype=np.uint8)
            accessory = np.dstack([accessory, alpha])
        return accessory
