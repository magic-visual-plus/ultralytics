from loguru import logger
import os 
# disable wd 
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO
pretrained_model_path = "/root/.cache/ckpts/yolov8m-seg.pt"
model = YOLO(pretrained_model_path)  # load a pretrained model (recommended for training)
lr = 0.01
train_base_dir = '/root/.cache/trains/2b75f2b4-cf67-463b-9ff6-58c534dfb52e'
# model.train(data=f'{train_base_dir}/dataset/data.yaml', device="cuda:1", copy_paste=0.0, mixup=0.0, epochs=200, project=f"{train_base_dir}/output", name="output", model=pretrained_model_path, lr0=lr, workers=1, batch=8, imgsz=1280)
model.train(data=f'{train_base_dir}/dataset/data.yaml', device="cuda:1", copy_paste=0.0, mixup=0.0, epochs=20, project=f"{train_base_dir}/output", name="output", model=pretrained_model_path, lr0=lr, workers=1, batch=8, imgsz=640)
