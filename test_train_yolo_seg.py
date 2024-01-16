from loguru import logger
import os 
# disable wd 
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO
pretrained_model_path = "/root/.cache/ckpts/yolov8m-seg.pt"
# pretrained_model_path = '/opt/ml/ultralytics/runs/train_seg/exp2/weights/best.pt'
# cfg_yaml = '/opt/ml/ultralytics/ultralytics/cfg/models/v8/yolov8m-seg-add-p2-head.yaml'
# model = YOLO(cfg_yaml)
# TODO enable later
# model.load(pretrained_model_path) # loading pretrain weights
model = YOLO(pretrained_model_path)  # load a pretrained model (recommended for training)
lr = 0.006
train_base_dir = '/root/.cache/trains/2b75f2b4-cf67-463b-9ff6-58c534dfb52e'
train_base_dir = '/root/.cache/trains/a4d98757-4f07-407c-9333-926a17b0b630/'
# model.train(data=f'{train_base_dir}/dataset/data.yaml', device="cuda:1", copy_paste=0.0, mixup=0.0, epochs=200, project=f"{train_base_dir}/output", name="output", model=pretrained_model_path, lr0=lr, workers=1, batch=8, imgsz=640)
model.train(data=f'{train_base_dir}/dataset/data.yaml', device="cuda:1", copy_paste=0.0, mixup=0.0, epochs=200, project=f"{train_base_dir}/output", name="output", model=pretrained_model_path, lr0=lr, optimizer='SGD', workers=0, batch=8, imgsz=1024)
# model.train(data=f'{train_base_dir}/dataset/data.yaml', device="cuda:1", copy_paste=0.0, mixup=0.0, epochs=20, project=f"{train_base_dir}/output", name="output", model=pretrained_model_path, lr0=lr, workers=1, batch=8, imgsz=640)
