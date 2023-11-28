from loguru import logger
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    pretrained_model_path = "/root/.cache/ckpts/yolov8m.pt"
    data_path = '/opt/ml/ultralytics/ultralytics/datasets/coco_1w.yaml'
    cfg_yaml = '/opt/ml/ultralytics/ultralytics/models/v8/yolov8m_custom.yaml'
    # cfg_yaml = '/opt/ml/ultralytics/ultralytics/models/v8/yolov8m-C2f-MSBlock.yaml'
    model = YOLO(cfg_yaml)
    model.load(pretrained_model_path) # loading pretrain weights
    lr = 0.001
    model.train(data=data_path,
                cache=False,
                imgsz=640,
                lr0=lr,
                epochs=100,
                batch=32,
                close_mosaic=10,
                workers=2,
                device='0,1',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train_ms',
                name='exp',
                )