from __future__ import annotations

"""
Static image height equalizer for "Justice for Short Kings".

Design goals
------------
1. Detect all people in the image with MediaPipe Pose (Tasks API).
2. Decide who is visually shortest / tallest in the frame.
   - For group photos where everyone stands roughly in one row, we use
     pixel height as the primary signal, which matches human perception.
3. Give the shortest person an option:
   - "stretch": scale them up in a *natural* way (no melted faces),
   - "hat" / "accessory": add a hat that makes their apparent height
     match the tallest person,
   - "both": small stretch + hat.
4. Make the stretching look clean by:
   - extracting a segmentation-based sprite of the person,
   - inpainting the original person out of the background,
   - re-compositing a uniformly scaled sprite back in.
"""

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

ArrayU8 = npt.NDArray[np.uint8]
ArrayF32 = npt.NDArray[np.float32]


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
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
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


@dataclass
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def clamp(self, frame_h: int, frame_w: int) -> "BoundingBox":
        left = int(np.clip(self.left, 0, frame_w - 1))
        right = int(np.clip(self.right, left + 1, frame_w))
        top = int(np.clip(self.top, 0, frame_h - 1))
        bottom = int(np.clip(self.bottom, top + 1, frame_h))
        return BoundingBox(left, top, right, bottom)


@dataclass
class PersonMeasurement:
    index: int
    pixel_landmarks: ArrayF32      # (33, 2)
    visibility: ArrayF32          # (33,)
    frame_size: Tuple[int, int]   # (H, W)
    segmentation_mask: Optional[ArrayF32] = None  # full-frame [0,1] mask

    def bounding_box(self, min_visibility: float = 0.2) -> BoundingBox:
        h, w = self.frame_size
        mask = self.visibility >= min_visibility
        if not np.any(mask):
            xs = self.pixel_landmarks[:, 0]
            ys = self.pixel_landmarks[:, 1]
        else:
            xs = self.pixel_landmarks[mask, 0]
            ys = self.pixel_landmarks[mask, 1]
        left = int(np.clip(xs.min(), 0, w - 1))
        right = int(np.clip(xs.max(), left + 1, w))
        top = int(np.clip(ys.min(), 0, h - 1))
        bottom = int(np.clip(ys.max(), top + 1, h))
        return BoundingBox(left, top, right, bottom)

    @property
    def height_px(self) -> float:
        """Visible height in pixels based on landmarks."""
        h, _ = self.frame_size
        head_y = self.head_top() or 0
        foot_y = self.feet_bottom() or (h - 1)
        return float(max(1, foot_y - head_y))

    def head_top(self) -> Optional[int]:
        h, _ = self.frame_size
        head_ids = [
            PoseLandmark.NOSE,
            PoseLandmark.LEFT_EYE,
            PoseLandmark.RIGHT_EYE,
            PoseLandmark.LEFT_EAR,
            PoseLandmark.RIGHT_EAR,
        ]
        ys = []
        for lid in head_ids:
            idx = lid.value
            if idx >= len(self.visibility):
                continue
            if self.visibility[idx] < 0.2:
                continue
            ys.append(self.pixel_landmarks[idx, 1])
        if not ys:
            return None
        return int(np.clip(min(ys), 0, h - 1))

    def feet_bottom(self) -> Optional[int]:
        h, _ = self.frame_size
        foot_ids = [
            PoseLandmark.LEFT_FOOT_INDEX,
            PoseLandmark.RIGHT_FOOT_INDEX,
            PoseLandmark.LEFT_HEEL,
            PoseLandmark.RIGHT_HEEL,
            PoseLandmark.LEFT_ANKLE,
            PoseLandmark.RIGHT_ANKLE,
        ]
        ys = []
        for lid in foot_ids:
            idx = lid.value
            if idx >= len(self.visibility):
                continue
            if self.visibility[idx] < 0.2:
                continue
            ys.append(self.pixel_landmarks[idx, 1])
        if not ys:
            return None
        return int(np.clip(max(ys), 0, h - 1))

    def shoulder_width(self) -> Optional[float]:
        pairs = [
            (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value),
            (PoseLandmark.LEFT_EAR.value, PoseLandmark.RIGHT_EAR.value),
        ]
        dists: List[float] = []
        for li, ri in pairs:
            if max(li, ri) >= len(self.visibility):
                continue
            if self.visibility[li] < 0.3 or self.visibility[ri] < 0.3:
                continue
            p_l = self.pixel_landmarks[li]
            p_r = self.pixel_landmarks[ri]
            dists.append(float(abs(p_l[0] - p_r[0])))
        if not dists:
            return None
        return float(np.median(dists))


@dataclass
class HeightAdjustmentResult:
    output_image: ArrayU8
    measurements: Sequence[PersonMeasurement]
    shortest_index: int
    reference_index: int
    method: str
    applied: bool


class HeightEqualizer:
    """Static-image equalizer for shortest/tallest people in a photo."""

    def __init__(
        self,
        pose_model_path: Path | str,
        accessory_path: Optional[Path | str] = None,
        *,
        max_people: int = 10,
        min_confidence: float = 0.3,
    ) -> None:
        pose_model_path = Path(pose_model_path)
        if not pose_model_path.exists():
            raise FileNotFoundError(pose_model_path)

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
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        self._accessory_rgba = self._load_accessory(accessory_path)
        if self._accessory_rgba is None:
            self._accessory_rgba = self._generate_default_hat()

    # ------------------------------------------------------------------ API

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
        self._landmarker = None

    # ----------------------------------------------------------------- core

    def evaluate_heights(self, image_bgr: ArrayU8) -> List[PersonMeasurement]:
        """Run MediaPipe Pose and collect per-person measurements."""
        if image_bgr.ndim != 3:
            raise ValueError("Expected a BGR image with shape (H, W, 3).")

        frame_h, frame_w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        pose_landmarks = result.pose_landmarks or []

        # Tasks API usually outputs *one* segmentation mask per frame,
        # so share it across all people.
        segmentation_full: Optional[ArrayF32] = None
        if result.segmentation_masks:
            segmentation_full = np.array(
                result.segmentation_masks[0].numpy_view(), dtype=np.float32
            )

        measurements: List[PersonMeasurement] = []

        for idx, lm_list in enumerate(pose_landmarks):
            xs = np.array([lm.x * frame_w for lm in lm_list], dtype=np.float32)
            ys = np.array([lm.y * frame_h for lm in lm_list], dtype=np.float32)
            pixel_landmarks = np.stack([xs, ys], axis=-1)  # (33, 2)
            visibility = np.array(
                [getattr(lm, "visibility", 1.0) for lm in lm_list], dtype=np.float32
            )

            measurements.append(
                PersonMeasurement(
                    index=idx,
                    pixel_landmarks=pixel_landmarks,
                    visibility=visibility,
                    frame_size=(frame_h, frame_w),
                    segmentation_mask=segmentation_full,
                )
            )

        return measurements

    def equalize(
        self,
        image_bgr: ArrayU8,
        *,
        method: str = "stretch",
        shortest_index: Optional[int] = None,
        reference_index: Optional[int] = None,
    ) -> HeightAdjustmentResult:
        """
        Equalize heights in a static image.

        Pipeline:
          1. Detect all people and measure their pixel heights.
          2. Find the visually shortest and tallest persons.
          3. Apply one of:
               - 'stretch': scale the shortest person up,
               - 'hat' / 'accessory': give them a hat,
               - 'both': small stretch + hat.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (H, W, 3).")

        working = image_bgr.copy()
        measurements = self.evaluate_heights(working)
        if not measurements:
            return HeightAdjustmentResult(
                output_image=working,
                measurements=(),
                shortest_index=-1,
                reference_index=-1,
                method=method,
                applied=False,
            )

        # 1) Decide who is shortest / tallest by *apparent* height in pixels.
        heights = np.array([m.height_px for m in measurements], dtype=np.float32)

        if shortest_index is None:
            shortest_index = int(np.argmin(heights))
        if reference_index is None:
            reference_index = int(np.argmax(heights))

        n = len(measurements)
        shortest_index = int(np.clip(shortest_index, 0, n - 1))
        reference_index = int(np.clip(reference_index, 0, n - 1))

        if shortest_index == reference_index:
            return HeightAdjustmentResult(
                output_image=working,
                measurements=measurements,
                shortest_index=shortest_index,
                reference_index=reference_index,
                method=method,
                applied=False,
            )

        shortest = measurements[shortest_index]
        tallest = measurements[reference_index]

        # How tall we want the shortest person to look (in pixels).
        # We also add a small bonus so they end up *slightly* taller than
        # the tallest, which makes the effect more visible.
        target_height_px = float(tallest.height_px) * 1.05
        current_height_px = float(shortest.height_px)

        if current_height_px <= 0:
            return HeightAdjustmentResult(
                output_image=working,
                measurements=measurements,
                shortest_index=shortest_index,
                reference_index=reference_index,
                method=method,
                applied=False,
            )

        method_norm = method.lower()
        output = working

        if method_norm == "stretch":
            output = self._stretch_sprite_to_height(output, shortest, target_height_px)
        elif method_norm in {"hat", "accessory"}:
            output = self._add_accessory_to_match_height(
                output, shortest, target_height_px
            )
        elif method_norm == "both":
            mid_height = (current_height_px + target_height_px) * 0.5
            temp = self._stretch_sprite_to_height(output, shortest, mid_height)
            output = self._add_accessory_to_match_height(
                temp, shortest, target_height_px
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return HeightAdjustmentResult(
            output_image=output,
            measurements=measurements,
            shortest_index=shortest_index,
            reference_index=reference_index,
            method=method_norm,
            applied=True,
        )

    # --------------------------------------------------------- sprite logic

    def _build_person_mask_full(self, person: PersonMeasurement) -> ArrayU8:
        """
        Build a full-frame 0/255 mask for this person.

        - Use MediaPipe segmentation if it's there.
        - Always OR it with a rectangular band that runs from head to feet
          (slightly padded), so we never chop the head off.
        """
        h, w = person.frame_size
        full = np.zeros((h, w), dtype=np.uint8)

        bbox = person.bounding_box()
        head_y = person.head_top() or bbox.top
        foot_y = person.feet_bottom() or bbox.bottom

        # --- rectangular fallback mask (guarantees full body, including head)
        pad_x = int(0.08 * bbox.width)  # 8% horizontal padding
        left = max(0, bbox.left - pad_x)
        right = min(w, bbox.right + pad_x)
        top = max(0, head_y - 5)
        bottom = min(h, foot_y + 3)

        rect_mask = np.zeros_like(full)
        rect_mask[top:bottom, left:right] = 255

        # --- segmentation mask (cropped to this person's bbox)
        if person.segmentation_mask is not None:
            seg = person.segmentation_mask
            seg_crop = seg[bbox.top:bbox.bottom, bbox.left:bbox.right]
            local = (seg_crop > 0.2).astype(np.uint8) * 255

            if local.sum() > 0:
                full[bbox.top:bbox.bottom, bbox.left:bbox.right] = local

        # Combine: use whichever has coverage (union of seg + rect)
        full = np.maximum(full, rect_mask)
        return full


    def _extract_sprite(self, image_bgr: ArrayU8, mask_full: ArrayU8):
        """Extract the tight sprite (image crop + mask crop) for a single person."""
        ys, xs = np.where(mask_full > 0)
        if ys.size == 0 or xs.size == 0:
            return None, None, None
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        left = int(xs.min())
        right = int(xs.max()) + 1

        sprite = image_bgr[top:bottom, left:right].copy()
        mask_crop = mask_full[top:bottom, left:right].copy()
        bbox = BoundingBox(left, top, right, bottom)
        return sprite, mask_crop, bbox

    def _stretch_sprite_to_height(
        self,
        image_bgr: ArrayU8,
        person: PersonMeasurement,
        target_height_px: float,
    ) -> ArrayU8:
        """
        More natural stretching:

        1) Build a segmentation-based sprite of the person.
        2) Inpaint the original person out of the background.
        3) Uniformly scale the sprite (with a small width increase).
        4) Re-composite it anchored at the feet.
        """
        frame_h, frame_w, _ = image_bgr.shape

        # Full-frame mask for this person (includes head/feet)
        mask_full = self._build_person_mask_full(person)
        sprite, mask_crop, bbox = self._extract_sprite(image_bgr, mask_full)
        if sprite is None:
            return image_bgr

        sprite_h, sprite_w, _ = sprite.shape
        if sprite_h <= 1 or sprite_w <= 1:
            return image_bgr

        current_height = float(sprite_h)
        desired_height = float(target_height_px)

        # Requested vertical scale
        scale_h = max(1.0, desired_height / current_height)

        # --- HARD CAP: don't let head go off-screen
        # Maximum height we can fit while keeping feet fixed is the
        # distance from the feet to the top of the frame.
        foot_y = person.feet_bottom()
        if foot_y is None:
            foot_y = bbox.bottom
        max_height_by_frame = max(20.0, float(foot_y))  # pixels from top to feet
        max_scale_frame = max_height_by_frame / current_height

        # Keep scale within both "requested" and "frame" bounds
        scale_h = float(np.clip(scale_h, 1.0, max_scale_frame * 0.98))

        # Small width growth so they don't look squashed
        scale_w = 1.0 + (scale_h - 1.0) * 0.3

        new_h = int(round(sprite_h * scale_h))
        new_w = int(round(sprite_w * scale_w))

        sprite_resized = cv2.resize(
            sprite, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )
        mask_resized = cv2.resize(
            mask_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # Inpaint the original person out of the background once
        bg = cv2.inpaint(image_bgr, mask_full, 3, cv2.INPAINT_TELEA)

        # Anchor at feet (same vertical foot position)
        new_bottom = int(np.clip(foot_y, 0, frame_h - 1))
        new_top = new_bottom - new_h

        # Horizontal position: center over original bbox center
        center_x = (bbox.left + bbox.right) // 2
        new_left = center_x - new_w // 2
        new_right = new_left + new_w

        # Clip sprite and compute source indices
        x1 = max(0, new_left)
        y1 = max(0, new_top)
        x2 = min(frame_w, new_right)
        y2 = min(frame_h, new_bottom)
        if x1 >= x2 or y1 >= y2:
            return image_bgr

        sx1 = x1 - new_left
        sy1 = y1 - new_top
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        sprite_crop = sprite_resized[sy1:sy2, sx1:sx2]
        mask_crop2 = mask_resized[sy1:sy2, sx1:sx2]

        # --- Feather the mask edges so it "diffuses" into background
        alpha = mask_crop2.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=3, sigmaY=3)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        fg = sprite_crop.astype(np.float32)
        roi = bg[y1:y2, x1:x2].astype(np.float32)

        blended = (fg * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        result = bg.copy()
        result[y1:y2, x1:x2] = blended
        return result

    # ---------------------------------------------------------- accessories

    def _add_accessory_to_match_height(
        self,
        image_bgr: ArrayU8,
        person: PersonMeasurement,
        target_height_px: float,
    ) -> ArrayU8:
        frame_h, frame_w, _ = image_bgr.shape
        extra = max(0.0, target_height_px - person.height_px)
        if extra <= 2.0:
            return image_bgr

        hat = self._accessory_rgba
        h, w, _ = hat.shape

        # Make the hat's visible height roughly equal to the height gap,
        # but clamp so it doesn't become ridiculous.
        scale = float(np.clip(extra / h, 0.6, 1.3))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        hat_resized = cv2.resize(hat, (new_w, new_h), interpolation=cv2.INTER_AREA)

        hat_rgb = hat_resized[:, :, :3]
        hat_a = hat_resized[:, :, 3].astype(np.float32) / 255.0

        # Place slightly above head_top.
        head_y = person.head_top()
        if head_y is None:
            bbox = person.bounding_box()
            head_y = bbox.top
        bottom_y = max(0, int(head_y - 0.03 * person.height_px))
        top_y = bottom_y - new_h

        # Horizontal center on eyes/ears if available, else bbox center.
        eye_ids = [PoseLandmark.LEFT_EYE.value, PoseLandmark.RIGHT_EYE.value]
        xs = []
        for idx in eye_ids:
            if idx < len(person.visibility) and person.visibility[idx] >= 0.2:
                xs.append(person.pixel_landmarks[idx, 0])
        if xs:
            head_x = int(np.mean(xs))
        else:
            bbox = person.bounding_box()
            head_x = (bbox.left + bbox.right) // 2

        left_x = head_x - new_w // 2

        # Clip
        x1 = max(0, left_x)
        y1 = max(0, top_y)
        x2 = min(frame_w, left_x + new_w)
        y2 = min(frame_h, top_y + new_h)
        if x1 >= x2 or y1 >= y2:
            return image_bgr

        sx1 = x1 - left_x
        sy1 = y1 - top_y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        hat_rgb_crop = hat_rgb[sy1:sy2, sx1:sx2]
        hat_a_crop = hat_a[sy1:sy2, sx1:sx2][..., None]

        roi = image_bgr[y1:y2, x1:x2].astype(np.float32)
        fg = hat_rgb_crop.astype(np.float32)

        blended = (fg * hat_a_crop + roi * (1.0 - hat_a_crop)).astype(np.uint8)
        result = image_bgr.copy()
        result[y1:y2, x1:x2] = blended
        return result

    # --------------------------------------------------------- accessories

    def _generate_default_hat(self) -> ArrayU8:
        width, height = 160, 120
        hat = np.zeros((height, width, 4), dtype=np.uint8)

        top_h = int(height * 0.6)
        cv2.rectangle(
            hat,
            (int(width * 0.2), 0),
            (int(width * 0.8), top_h),
            (30, 30, 30, 255),
            thickness=-1,
        )
        cv2.rectangle(
            hat,
            (0, top_h - 8),
            (width, height),
            (20, 20, 20, 255),
            thickness=-1,
        )
        cv2.rectangle(
            hat,
            (int(width * 0.22), int(top_h * 0.35)),
            (int(width * 0.78), int(top_h * 0.5)),
            (220, 60, 160, 255),
            thickness=-1,
        )
        return hat

    def _load_accessory(self, accessory_path: Optional[Path | str]) -> Optional[ArrayU8]:
        if accessory_path is None:
            return None
        p = Path(accessory_path)
        if not p.exists():
            raise FileNotFoundError(f"Accessory image not found: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read accessory image: {p}")
        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            img = np.dstack([img, alpha])
        return img
