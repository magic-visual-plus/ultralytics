# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, predict
from .train import DetectionTrainer, train
from .val import DetectionValidator, val
from .test import DetectionTestValidator, test

__all__ = 'DetectionPredictor', 'predict', 'DetectionTrainer', 'train', 'DetectionValidator', 'val', DetectionTestValidator, test
