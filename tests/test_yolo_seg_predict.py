from loguru import logger
import os 
# disable wd 
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO

# model = YOLO('/opt/product/test_datas/test_yolo_seg/output/weights/best.pt')  # load a pretrained model (recommended for training)
model = YOLO("/root/.cache/ckpts/yolov8n-seg.pt")
metrics = model.test(data="/opt/product/test_datas/test_yolo_seg/dataset/data.yaml", split="test", conf=.4)
logger.info("metrics {}", metrics)