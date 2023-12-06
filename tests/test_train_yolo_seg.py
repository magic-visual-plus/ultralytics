from loguru import logger
import os 
# disable wd 
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO
pretrained_model_path = "/root/.cache/ckpts/yolov8m-seg.pt"
model = YOLO(pretrained_model_path)  # load a pretrained model (recommended for training)
lr = 0.01
train_base_dir = '/opt/product/test_datas/seg_1205_more'
model.train(data=f'{train_base_dir}/dataset/data.yaml', device="cuda:1", copy_paste=0.0, mixup=0.0, epochs=80, project=f"{train_base_dir}/output", name="output", model=pretrained_model_path, lr0=lr, workers=1, batch=8, imgsz=1280)
