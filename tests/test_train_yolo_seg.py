from loguru import logger
import os 
# disable wd 
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO
pretrained_model_path = "/root/.cache/ckpts/yolov8n-seg.pt"
model = YOLO(pretrained_model_path)  # load a pretrained model (recommended for training)
lr = 0.01
train_base_dir = '/opt/ml/ultralytics/datas'
model.train(data='/opt/ml/ultralytics/datas/data.yaml', epochs=100, project=f"{train_base_dir}/output", name="output", model=pretrained_model_path, lr0=lr, workers=2)
