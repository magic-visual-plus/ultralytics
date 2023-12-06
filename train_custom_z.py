from loguru import logger
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    pretrained_model_path = "/root/.cache/ckpts/yolov8m.pt"
    pretrained_model_path = '/opt/ml/ultralytics/runs/train_ms/exp3/weights/best.pt'
    data_path = '/opt/ml/ultralytics/ultralytics/datasets/coco_1w.yaml'    
    # data_path = '/opt/product/test_datas/yolo_1127/dataset/data.yaml'
    cfg_yaml = '/opt/ml/ultralytics/ultralytics/models/v8/yolov8m_custom_z.yaml'
    # cfg_yaml = '/opt/ml/ultralytics/ultralytics/models/v8/yolov8m-C2f-MSBlock.yaml'
    # cfg_yaml = '/opt/ml/yolov8-main/ultralytics/cfg/models/v8/yolov8m-C2f-RFCAConv.yaml'
    # cfg_yaml = '/opt/ml/yolov8-main/ultralytics/cfg/models/v8/yolov8m-C2f-DCNV3.yaml'
    model = YOLO(cfg_yaml)
    model.load(pretrained_model_path) # loading pretrain weights
    lr = 0.001
    model.train(data=data_path,
                cache=False,
                imgsz=640,
                lr0=lr,
                epochs=20,
                batch=32,
                close_mosaic=10,
                workers=2,
                device='0,1',
                optimizer='SGD', # using SGD
                # resume='/opt/ml/ultralytics/runs/train_ms/exp/weights/last.pt',
                # resume='/opt/ml/yolov8-main/runs/train_ms/exp56/weights/last.pt', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train_ms',
                name='exp',
                )