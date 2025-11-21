"""
Will's Segmentation Expansion Feature
Makes the shortest person in an image taller using segmentation and mask expansion.

V1: SAM auto-segmentation with filtering
V2: YOLO + SAM (2-stage instance segmentation) - RECOMMENDED
"""

from .segmentation_expander import SegmentationExpander, PersonMask
from .segmentation_expander_v2 import SegmentationExpanderV2, PersonInstance

__all__ = [
    "SegmentationExpander",
    "PersonMask",
    "SegmentationExpanderV2",
    "PersonInstance"
]
