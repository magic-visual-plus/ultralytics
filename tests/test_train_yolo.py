from loguru import logger
import pytest
from ultralytics import YOLO


# def freeze_layer(trainer):
#     model = trainer.model
#     num_freeze = 10
#     logger.info(f"Freezing {num_freeze} layers")
#     freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
#     for k, v in model.named_parameters(): 
#         v.requires_grad = True  # train all layers 
#         if any(x in k for x in freeze): 
#             logger.info(f'freezing {k}') 
#             v.requires_grad = False 
#     logger.info(f"{num_freeze} layers are freezed.")
    
# def print_layer(trainer):
#     model = trainer.model
#     freeze = [f'model.{x}.' for x in range(10)]  # layers to freeze 
#     for k, v in model.named_parameters():
#         if any(x in k for x in freeze):
#             logger.info(k, v.requires_grad)

def call_on_val_end(validitor):
    logger.info("on_val_end")    

def call_on_train_end(trainer):
    logger.info("train stoped, best epoch is {}" , trainer.stopper.best_epoch)

pretrained_model_path = "/root/.cache/ckpts/yolov8m.pt"
pretrained_model_path = '/opt/product/test_datas/yolo_1125/dataset/yolo_more_neck.pt'
model = YOLO("/opt/ml/ultralytics/ultralytics/models/v8/yolov8m_custom.yaml").load(pretrained_model_path)
# model = YOLO(pretrained_model_path)  # load a pretrained model (recommended for training)

model.add_callback("on_train_end", call_on_train_end)
model.add_callback("on_val_end", call_on_val_end)
lr = 0.005
# freeze = 10
train_base_dir = '/opt/product/test_datas/yolo_chip_v2'
train_base_dir = '/opt/product/test_datas/yolo_1125'
model.train(data=f'{train_base_dir}/dataset/data.yaml', epochs=10, project=f"{train_base_dir}/output", name="output", batch=8, model=pretrained_model_path, lr0=lr, workers=1, device='1')
