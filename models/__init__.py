"""Models package for object detection."""

from .detector import ObjectDetector
from .losses import DetectionLoss, IoULoss

__all__ = ['ObjectDetector', 'DetectionLoss', 'IoULoss']
