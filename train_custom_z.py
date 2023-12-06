from loguru import logger
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    pretrained_model_path = "/root/.cache/ckpts/yolov8m.pt"
    # DCNv3
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp5/weights/best.pt'
    
    # pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp/weights/best.pt'
    # pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp3/weights/best.pt'
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/dcn_v3_zero_exp13/weights/best.pt'
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp20/weights/best.pt'
    
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/dcn_v3_more_head/weights/last.pt'
    
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp21/weights/best.pt'
    
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp56_custom_z/weights/best.pt'
    
    # data_path = '/opt/ml/ultralytics/ultralytics/datasets/coco_1w.yaml'
    # data_path = '/opt/product/test_datas/yolo_1125/dataset/data.yaml'
    
    data_path = '/opt/product/test_datas/yolo_1127/dataset/data.yaml'
    pretrained_model_path = "/opt/ml/ultralytics/runs/train_ms/exp2/weights/best.pt"
    cfg_yaml = '/opt/ml/ultralytics/ultralytics/models/v8/yolov8m_custom_z.yaml'
    model = YOLO(cfg_yaml)
    model.load(pretrained_model_path) # loading pretrain weights
    lr = 0.01
    model.train(data=data_path,
                cache=False,
                imgsz=640,
                lr0=lr,
                epochs=200,
                batch=16,
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