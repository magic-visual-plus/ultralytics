from loguru import logger
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # coco path
    data_path = '/opt/ml/new_ml/ultralytics/ultralytics/cfg/datasets/coco_seg_1w.yaml'
    pretrained_model_path = "/opt/ml/ultralytics/runs/train_seg/exp2/weights/best.pt"
    # yolo seg m cfg
    cfg_yaml = '/opt/ml/ultralytics/ultralytics/cfg/models/v8/yolov8m-seg-add-p2-head.yaml'
    model = YOLO(cfg_yaml)
    # TODO enable later
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
                # resume='/opt/ml/ultralytics/runs/train_seg/exp/weights/last.pt',
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train_seg',
                name='exp',
                )