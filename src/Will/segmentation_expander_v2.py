"""
Segmentation Expansion Feature v2 - 2-Stage Instance Segmentation
Uses YOLO for person detection + SAM for precise segmentation

Author: Will
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
from PIL import Image
import warnings
import os
warnings.filterwarnings('ignore')

# Allow loading non-safetensors weights (bypass torch.load security check)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Lazy imports for diffusion models
try:
    from diffusers import AutoPipelineForInpainting, ControlNetModel, StableDiffusionControlNetInpaintPipeline
    from controlnet_aux import OpenposeDetector
    DIFFUSERS_AVAILABLE = True
    print("✓ Diffusers imported successfully")
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    print(f"⚠ Diffusers import failed: {e}")


@dataclass
class PersonInstance:
    """Represents a detected person with bounding box and segmentation."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) - XYXY format
    mask: Optional[np.ndarray]  # Binary mask (H, W)
    confidence: float  # Detection confidence
    height: float  # Height in pixels (y2 - y1)
    width: float  # Width in pixels (x2 - x1)
    center_x: int  # Center X coordinate
    center_y: int  # Center Y coordinate
    index: int  # Person index


class SegmentationExpanderV2:
    """
    2-Stage Instance Segmentation for Height Expansion.

    Stage 1: YOLO detects people and provides bounding boxes
    Stage 2: SAM segments each person using bounding box prompts
    """

    def __init__(self,
                 sam_checkpoint: Optional[str] = None,
                 sam_model_type: str = "vit_h",
                 yolo_model: str = "yolov8n.pt",
                 device: str = "cpu"):
        """
        Initialize the 2-stage segmentation system.

        Args:
            sam_checkpoint: Path to SAM model checkpoint
            sam_model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            yolo_model: YOLO model name or path
            device: Device to run on ("cpu" or "cuda")
        """
        self.device = device

        # Initialize YOLO for person detection
        print(f"Loading YOLO model: {yolo_model}...")
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model)
        print("✓ YOLO model loaded!")

        # Initialize SAM for segmentation
        if sam_checkpoint and Path(sam_checkpoint).exists():
            print(f"Loading SAM model from {sam_checkpoint}...")
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
            self.sam_available = True
            print("✓ SAM model loaded!")
        else:
            print("⚠ SAM checkpoint not found. Box-based segmentation will be used.")
            self.sam_available = False
            self.sam_predictor = None

        # Diffusion models (lazy loaded)
        self.inpaint_pipe = None
        self.inpaint_method = None
        self.pose_detector = None

    def initialize_replacement_models(self, method="sdxl"):
        """
        Lazy initialization of diffusion models for person replacement.

        Args:
            method: "sdxl" for SDXL Turbo or "controlnet" for ControlNet
        """
        if not DIFFUSERS_AVAILABLE:
            print("❌ Diffusers not installed. Person replacement unavailable.")
            return False

        # Check if already initialized
        if self.inpaint_pipe is not None and self.inpaint_method == method:
            return True

        try:
            print(f"\n{'='*60}")
            print(f"INITIALIZING {method.upper()} REPLACEMENT MODEL")
            print(f"{'='*60}")
            print("⏳ Downloading models (first time only, ~4-6GB)...")

            device = self.device
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"

            dtype = torch.float16 if device != "cpu" else torch.float32

            if method == "sdxl":
                self.inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    torch_dtype=dtype,
                    variant="fp16" if device != "cpu" else None
                ).to(device)

            elif method == "controlnet":
                # Load ControlNet (SD 1.5 compatible)
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_openpose",
                    torch_dtype=dtype
                )

                # Use original SD inpainting model with ControlNet
                self.inpaint_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    safety_checker=None
                ).to(device)

                self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

            self.inpaint_method = method
            print(f"✓ {method.upper()} model loaded!")
            return True

        except Exception as e:
            print(f"❌ Error loading {method} model: {e}")
            return False

    def replace_person_with_prompt(self, image: np.ndarray, person: PersonInstance,
                                    prompt: str, method="sdxl") -> np.ndarray:
        """
        Replace a person with AI-generated content.

        Args:
            image: Original image (BGR)
            person: PersonInstance to replace
            prompt: Text description (e.g., "a kangaroo", "ghost")
            method: "sdxl" or "controlnet"

        Returns:
            Image with person replaced (BGR)
        """
        if not self.initialize_replacement_models(method):
            print("⚠ Replacement unavailable, returning original")
            return image

        print(f"\n{'='*60}")
        print(f"REPLACING PERSON {person.index} WITH: '{prompt}'")
        print(f"{'='*60}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray((person.mask * 255).astype(np.uint8))

        try:
            if method == "sdxl":
                result_pil = self.inpaint_pipe(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=4,
                    strength=0.99,
                    guidance_scale=0.0
                ).images[0]

            elif method == "controlnet":
                x1, y1, x2, y2 = person.bbox
                person_crop = image_rgb[y1:y2, x1:x2]
                person_pil = Image.fromarray(person_crop)
                pose_map = self.pose_detector(person_pil)

                pose_full = Image.new('RGB', image_pil.size, (0, 0, 0))
                pose_full.paste(pose_map, (x1, y1))

                result_pil = self.inpaint_pipe(
                    prompt=prompt + ", high quality, detailed",
                    negative_prompt="blurry, distorted, low quality",
                    image=image_pil,
                    mask_image=mask_pil,
                    control_image=pose_full,
                    num_inference_steps=20,
                    controlnet_conditioning_scale=0.8,
                    guidance_scale=7.5
                ).images[0]

            result_np = np.array(result_pil)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            print(f"✓ Replacement complete!")
            return result_bgr

        except Exception as e:
            print(f"❌ Error during replacement: {e}")
            return image

    def detect_people_yolo(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[PersonInstance]:
        """
        Stage 1: Detect people using YOLO.

        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections

        Returns:
            List of PersonInstance objects (without masks yet)
        """
        print("\n" + "=" * 60)
        print("STAGE 1: YOLO PERSON DETECTION")
        print("=" * 60)

        import time
        start_time = time.time()

        # Run YOLO detection (class 0 = person in COCO dataset)
        results = self.yolo(image, classes=[0], conf=conf_threshold, verbose=False)

        elapsed = time.time() - start_time
        print(f"✓ YOLO detection completed in {elapsed:.2f} seconds")

        people = []
        for idx, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            confidence = float(conf.cpu().numpy())

            height = y2 - y1
            width = x2 - x1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            person = PersonInstance(
                bbox=(x1, y1, x2, y2),
                mask=None,  # Will be filled in stage 2
                confidence=confidence,
                height=float(height),
                width=float(width),
                center_x=center_x,
                center_y=center_y,
                index=idx
            )
            people.append(person)

        # Sort by height (tallest first)
        people.sort(key=lambda p: p.height, reverse=True)

        # Update indices after sorting
        for idx, person in enumerate(people):
            person.index = idx

        print(f"✓ Detected {len(people)} people")
        if len(people) > 0:
            print(f"  Height range: {people[-1].height:.0f}px to {people[0].height:.0f}px")

        return people

    def segment_people_sam(self, image_rgb: np.ndarray, people: List[PersonInstance]) -> List[PersonInstance]:
        """
        Stage 2: Segment each person using SAM with bounding box prompts.

        Args:
            image_rgb: Input image in RGB format
            people: List of PersonInstance objects from YOLO

        Returns:
            Updated list with segmentation masks
        """
        if not self.sam_available or self.sam_predictor is None:
            print("\n⚠ SAM not available, using bounding box masks instead")
            # Fallback: use bounding boxes as masks
            h, w = image_rgb.shape[:2]
            for person in people:
                mask = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = person.bbox
                mask[y1:y2, x1:x2] = True
                person.mask = mask
            return people

        print("\n" + "=" * 60)
        print("STAGE 2: SAM SEGMENTATION WITH BOX PROMPTS")
        print("=" * 60)

        import time
        start_time = time.time()

        # Set image for SAM
        self.sam_predictor.set_image(image_rgb)
        print(f"Processing {len(people)} people with SAM...")

        # Segment each person using their bounding box as prompt
        for idx, person in enumerate(people):
            x1, y1, x2, y2 = person.bbox

            # SAM expects box in XYXY format (same as YOLO)
            box_prompt = np.array([x1, y1, x2, y2])

            # Get mask from SAM
            masks, scores, logits = self.sam_predictor.predict(
                box=box_prompt,
                multimask_output=False  # Single mask per box
            )

            # Take the first (and only) mask
            person.mask = masks[0]

            if (idx + 1) % 5 == 0 or idx == len(people) - 1:
                print(f"  Processed {idx + 1}/{len(people)} people...")

        elapsed = time.time() - start_time
        print(f"✓ SAM segmentation completed in {elapsed:.2f} seconds")
        print(f"✓ All {len(people)} people segmented")

        return people

    def expand_person(self,
                     image: np.ndarray,
                     person: PersonInstance,
                     target_height: float) -> Tuple[np.ndarray, PersonInstance]:
        """
        Scale a person to match target height (proportional scaling).
        Handles both expansion (scale > 1) and shrinking (scale < 1).

        Args:
            image: Original image (BGR)
            person: PersonInstance to scale
            target_height: Target height in pixels

        Returns:
            Tuple of (result_image, scaled_person_instance)
        """
        h, w = image.shape[:2]
        scale_factor = target_height / person.height
        is_shrinking = scale_factor < 1.0

        print("\n" + "=" * 60)
        if is_shrinking:
            print("STAGE 3: SHRINKING PERSON")
        else:
            print("STAGE 3: EXPANDING PERSON")
        print("=" * 60)
        print(f"  Original height: {person.height:.0f}px")
        print(f"  Target height: {target_height:.0f}px")
        print(f"  Scale factor: {scale_factor:.2f}x")
        print(f"  Maintaining aspect ratio...")

        # Extract person region
        x1, y1, x2, y2 = person.bbox
        person_img = image[y1:y2, x1:x2].copy()
        person_mask = person.mask[y1:y2, x1:x2].copy()

        # Calculate new dimensions (proportional scaling)
        new_height = int(person.height * scale_factor)
        new_width = int(person.width * scale_factor)

        print(f"  Original size: {int(person.width)}x{int(person.height)}px")
        print(f"  New size: {new_width}x{new_height}px")

        # Resize person and mask
        person_expanded = cv2.resize(person_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        mask_expanded = cv2.resize(person_mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        mask_expanded = mask_expanded.astype(bool)

        # Create result image
        result_image = image.copy()

        # Calculate position (keep center_x, align bottom)
        new_center_x = person.center_x
        new_bottom_y = y2
        new_top_y = max(0, new_bottom_y - new_height)
        new_left_x = max(0, min(w - new_width, new_center_x - new_width // 2))
        new_right_x = min(w, new_left_x + new_width)

        # Clip if out of bounds
        src_y_start = 0 if new_top_y >= 0 else -new_top_y
        src_y_end = new_height if new_bottom_y <= h else new_height - (new_bottom_y - h)
        dst_y_start = max(0, new_top_y)
        dst_y_end = min(h, new_bottom_y)

        src_x_start = 0 if new_left_x >= 0 else -new_left_x
        src_x_end = new_width if new_right_x <= w else new_width - (new_right_x - w)
        dst_x_start = max(0, new_left_x)
        dst_x_end = min(w, new_right_x)

        if is_shrinking:
            # For shrinking: First place the smaller person, then inpaint exposed areas
            print("  Compositing scaled person...")
            person_slice = person_expanded[src_y_start:src_y_end, src_x_start:src_x_end]
            mask_slice = mask_expanded[src_y_start:src_y_end, src_x_start:src_x_end]

            # Feather edges
            mask_slice_uint8 = mask_slice.astype(np.uint8) * 255
            mask_slice_feathered = cv2.GaussianBlur(mask_slice_uint8, (5, 5), 0)
            alpha = mask_slice_feathered.astype(float) / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)

            # Blend scaled person onto image
            dst_region = result_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
            blended = (person_slice * alpha + dst_region * (1 - alpha)).astype(np.uint8)
            result_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended

            # Calculate exposed area (original mask minus new scaled mask)
            print("  Inpainting exposed areas around scaled person...")
            new_mask = np.zeros((h, w), dtype=bool)
            new_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = mask_slice

            # Exposed area = original mask AND NOT new mask
            exposed_mask = person.mask & ~new_mask
            exposed_mask_uint8 = exposed_mask.astype(np.uint8) * 255

            # Inpaint the exposed areas
            result_image = cv2.inpaint(result_image, exposed_mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        else:
            # For expansion: Inpaint original location first, then place expanded person
            print("  Inpainting original location...")
            original_mask_uint8 = person.mask.astype(np.uint8) * 255
            result_image = cv2.inpaint(result_image, original_mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Place expanded person with alpha blending
            print("  Compositing expanded person...")
            person_slice = person_expanded[src_y_start:src_y_end, src_x_start:src_x_end]
            mask_slice = mask_expanded[src_y_start:src_y_end, src_x_start:src_x_end]

            # Feather edges
            mask_slice_uint8 = mask_slice.astype(np.uint8) * 255
            mask_slice_feathered = cv2.GaussianBlur(mask_slice_uint8, (5, 5), 0)
            alpha = mask_slice_feathered.astype(float) / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)

            # Blend
            dst_region = result_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
            blended = (person_slice * alpha + dst_region * (1 - alpha)).astype(np.uint8)
            result_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended

        # Create expanded person instance
        expanded_mask = np.zeros((h, w), dtype=bool)
        expanded_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = mask_slice

        expanded_person = PersonInstance(
            bbox=(dst_x_start, dst_y_start, dst_x_end, dst_y_end),
            mask=expanded_mask,
            confidence=person.confidence,
            height=float(dst_y_end - dst_y_start),
            width=float(dst_x_end - dst_x_start),
            center_x=(dst_x_start + dst_x_end) // 2,
            center_y=(dst_y_start + dst_y_end) // 2,
            index=person.index
        )

        if is_shrinking:
            print(f"✓ Shrinking complete!")
        else:
            print(f"✓ Expansion complete!")

        return result_image, expanded_person

    def visualize_detections(self, image: np.ndarray, people: List[PersonInstance], title: str = "Detections") -> np.ndarray:
        """Visualize people with bounding boxes."""
        vis = image.copy()

        for person in people:
            x1, y1, x2, y2 = person.bbox
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"Person {person.index}: {person.height:.0f}px"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return vis

    def level_people(self, image: np.ndarray, people: List[PersonInstance]) -> Tuple[np.ndarray, List[PersonInstance]]:
        """
        Align all people to the same baseline (ground level).

        Args:
            image: Input image (BGR)
            people: List of PersonInstance objects

        Returns:
            Tuple of (result_image, updated_people_list)
        """
        h, w = image.shape[:2]
        result_image = image.copy()

        # Find the common baseline (lowest bottom y-coordinate)
        baseline_y = max(person.bbox[3] for person in people)  # bbox is (x1, y1, x2, y2)

        print(f"\n{'='*60}")
        print(f"LEVELING PEOPLE TO COMMON BASELINE")
        print(f"{'='*60}")
        print(f"  Baseline Y: {baseline_y}px")

        updated_people = []

        for person in people:
            x1, y1, x2, y2 = person.bbox
            current_bottom_y = y2
            offset_y = baseline_y - current_bottom_y

            if abs(offset_y) < 1:  # Already at baseline
                print(f"  Person {person.index}: Already at baseline")
                updated_people.append(person)
                continue

            print(f"  Person {person.index}: Moving {offset_y:+.0f}px (from y={current_bottom_y} to y={baseline_y})")

            # Calculate new position
            new_y1 = y1 + offset_y
            new_y2 = y2 + offset_y

            # Ensure within bounds
            if new_y1 < 0:
                new_y1 = 0
                new_y2 = new_y1 + (y2 - y1)
            if new_y2 > h:
                new_y2 = h
                new_y1 = new_y2 - (y2 - y1)

            # Extract person image and mask
            person_img = image[y1:y2, x1:x2].copy()
            person_mask_crop = person.mask[y1:y2, x1:x2].copy()

            # Inpaint original location
            original_mask_uint8 = person.mask.astype(np.uint8) * 255
            result_image = cv2.inpaint(result_image, original_mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            # Place person at new location with alpha blending
            person_height = y2 - y1
            person_width = x2 - x1

            # Feather edges
            mask_uint8 = person_mask_crop.astype(np.uint8) * 255
            mask_feathered = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
            alpha = mask_feathered.astype(float) / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)

            # Blend at new location
            dst_y1 = int(new_y1)
            dst_y2 = int(new_y2)
            dst_region = result_image[dst_y1:dst_y2, x1:x2]
            blended = (person_img * alpha + dst_region * (1 - alpha)).astype(np.uint8)
            result_image[dst_y1:dst_y2, x1:x2] = blended

            # Create updated person instance
            new_mask = np.zeros((h, w), dtype=bool)
            new_mask[dst_y1:dst_y2, x1:x2] = person_mask_crop

            updated_person = PersonInstance(
                bbox=(x1, dst_y1, x2, dst_y2),
                mask=new_mask,
                confidence=person.confidence,
                height=float(dst_y2 - dst_y1),
                width=float(x2 - x1),
                center_x=person.center_x,
                center_y=(dst_y1 + dst_y2) // 2,
                index=person.index
            )
            updated_people.append(updated_person)

        print(f"✓ Leveling complete!")
        return result_image, updated_people

    def visualize_segmentations(self, image: np.ndarray, people: List[PersonInstance],
                                shortest_idx: Optional[int] = None) -> np.ndarray:
        """Visualize segmentation masks with colors."""
        vis = image.copy()
        overlay = image.copy()

        for person in people:
            if person.mask is None:
                continue

            # Color: Red for shortest, Green for others
            if shortest_idx is not None and person.index == shortest_idx:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green

            # Apply colored mask
            overlay[person.mask] = color

            # Draw bounding box
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Blend
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        return vis

    def process_image(self, image: np.ndarray, conf_threshold: float = 0.5, equalize: bool = False,
                     replace: bool = False, replace_prompt: str = "", replace_method: str = "sdxl") -> Dict[str, np.ndarray]:
        """
        Main processing pipeline.

        Args:
            image: Input image (BGR)
            conf_threshold: YOLO confidence threshold
            equalize: If True, scale all people to average height; if False, expand shortest to tallest
            replace: If True, replace shortest person with AI-generated content
            replace_prompt: Text description for replacement (e.g., "a kangaroo")
            replace_method: "sdxl" or "controlnet"

        Returns:
            Dictionary with visualization images at each stage
        """
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = {
            'step1_original': image.copy(),
            'step2_detections': None,
            'step3_segmentations': None,
            'step4_expansion': None,
            'step5_final': None
        }

        # Stage 1: YOLO Detection
        people = self.detect_people_yolo(image, conf_threshold)

        if len(people) == 0:
            print("\n⚠ No people detected!")
            return {k: image for k in results.keys()}

        # Visualize detections
        results['step2_detections'] = self.visualize_detections(image, people)

        print(f"\n{'='*60}")
        print(f"DETECTED {len(people)} PEOPLE")
        print(f"{'='*60}")
        for person in people:
            print(f"  Person {person.index}: {person.height:.0f}px (conf: {person.confidence:.2f})")
        print(f"{'='*60}")

        # Stage 2: SAM Segmentation
        people = self.segment_people_sam(image_rgb, people)

        # Find shortest and tallest
        shortest_person = people[-1]
        tallest_person = people[0]

        # Calculate average height for equalize mode
        average_height = sum(p.height for p in people) / len(people)

        print(f"\n{'='*60}")
        print(f"HEIGHT ANALYSIS")
        print(f"{'='*60}")
        print(f"  Shortest: Person {shortest_person.index} ({shortest_person.height:.0f}px)")
        print(f"  Tallest: Person {tallest_person.index} ({tallest_person.height:.0f}px)")
        print(f"  Average: {average_height:.0f}px")
        print(f"  Difference: {tallest_person.height - shortest_person.height:.0f}px")
        if equalize:
            print(f"  Mode: EQUALIZE - scaling all people to average height")
        else:
            print(f"  Mode: EXPAND - expanding shortest to tallest")
            print(f"  Expansion needed: {(tallest_person.height / shortest_person.height - 1) * 100:.1f}%")
        print(f"{'='*60}")

        # Visualize segmentations
        results['step3_segmentations'] = self.visualize_segmentations(image, people, shortest_person.index)

        # Stage 3: Expand/scale people
        if equalize:
            # Equalize mode: Scale all people to average height
            expanded_image = image.copy()
            expansion_vis = expanded_image.copy()
            overlay = expanded_image.copy()

            for person in people:
                if abs(person.height - average_height) > 1:  # Only process if difference is significant
                    expanded_image, scaled_person = self.expand_person(expanded_image, person, average_height)

                    # Visualize with different colors
                    if scaled_person.mask is not None:
                        if person.height < average_height:
                            overlay[scaled_person.mask] = (255, 0, 0)  # Blue for expanded
                        else:
                            overlay[scaled_person.mask] = (0, 255, 255)  # Yellow for shrunk
                        x1, y1, x2, y2 = scaled_person.bbox
                        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                    (255, 0, 0) if person.height < average_height else (0, 255, 255), 2)

            expansion_vis = cv2.addWeighted(expanded_image, 0.6, overlay, 0.4, 0)
            results['step4_expansion'] = expansion_vis
            results['step5_final'] = expanded_image

        else:
            # Original mode: Expand shortest person to tallest
            if shortest_person.height < tallest_person.height:
                expanded_image, expanded_person = self.expand_person(image, shortest_person, tallest_person.height)

                # Visualize expansion (show expanded person with blue mask)
                expansion_vis = expanded_image.copy()
                overlay = expanded_image.copy()
                if expanded_person.mask is not None:
                    overlay[expanded_person.mask] = (255, 0, 0)  # Blue
                    x1, y1, x2, y2 = expanded_person.bbox
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                expansion_vis = cv2.addWeighted(expansion_vis, 0.6, overlay, 0.4, 0)

                results['step4_expansion'] = expansion_vis
                results['step5_final'] = expanded_image
            else:
                print("\n⚠ Shortest person is already tallest - no expansion needed")
                results['step4_expansion'] = image.copy()
                results['step5_final'] = image.copy()

        # Optional: Replace shortest person with AI-generated content
        if replace and replace_prompt and len(people) > 0:
            final_image = results['step5_final']
            shortest_person = people[-1]  # Already sorted by height

            print(f"\n{'='*60}")
            print(f"AI REPLACEMENT MODE")
            print(f"{'='*60}")
            print(f"  Target: Person {shortest_person.index} (shortest)")
            print(f"  Prompt: '{replace_prompt}'")
            print(f"  Method: {replace_method}")

            final_image = self.replace_person_with_prompt(
                final_image, shortest_person, replace_prompt, replace_method
            )
            results['step5_final'] = final_image

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*60}")

        return results

    def display_results(self, results: Dict[str, np.ndarray], max_height: int = 800):
        """Display results in separate windows."""
        print("\n" + "=" * 60)
        print("DISPLAYING RESULTS")
        print("=" * 60)

        steps = [
            ('step1_original', '1. Original Image'),
            ('step2_detections', '2. YOLO Detections'),
            ('step3_segmentations', '3. SAM Segmentations'),
            ('step4_expansion', '4. Expansion (Blue = Expanded)'),
            ('step5_final', '5. Final Result')
        ]

        for idx, (key, title) in enumerate(steps):
            if key not in results or results[key] is None:
                continue

            img = results[key]
            h, w = img.shape[:2]

            # Resize if too large
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                img = cv2.resize(img, (new_w, max_height))

            # Add title
            label_img = img.copy()
            cv2.putText(label_img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(label_img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, label_img)

            print(f"  Showing: {title}")
            cv2.waitKey(300)  # Brief delay between windows

        print("\n" + "=" * 60)
        print("All windows displayed!")
        print("Press any key in any window to close all...")
        print("=" * 60)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="2-Stage Segmentation Expansion Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--sam-checkpoint", type=str, required=True,
                       help="Path to SAM checkpoint")
    parser.add_argument("--sam-type", type=str, default="vit_b",
                       choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                       help="YOLO model (n/s/m/l/x)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"])

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    print(f"Loaded image: {image.shape}")

    # Initialize system
    expander = SegmentationExpanderV2(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_type,
        yolo_model=args.yolo_model,
        device=args.device
    )

    # Process
    results = expander.process_image(image, conf_threshold=args.conf)

    # Display
    expander.display_results(results)


if __name__ == "__main__":
    main()
