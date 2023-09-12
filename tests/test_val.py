from loguru import logger
import pytest
from ultralytics import YOLO

model = YOLO('/tmp/trains/710/output/best.pt')  # load an official model        
# Validate the model
metrics = model.val(data="/tmp/trains/710/dataset/data.yaml", split="test")
logger.info(metrics)