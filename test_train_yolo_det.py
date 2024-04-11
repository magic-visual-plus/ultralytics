from loguru import logger
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


def train_yolo_custom():
    pretrained_model_path = "/root/.cache/ckpts/yolov8m.pt"
    # DCNv3
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp5/weights/best.pt'
    pretrained_model_path = '/opt/ml/yolov8-main/runs/train_ms/exp56_custom_z/weights/best.pt'
    
    # data_path = '/opt/ml/ultralytics/ultralytics/datasets/coco_1w.yaml'
    # data_path = '/opt/product/test_datas/yolo_1125/dataset/data.yaml'
    data_path = '/opt/product/test_datas/yolo_1127/dataset/data.yaml'
    # data_path = '/root/.cache/trains/c589deeb-7349-4ed8-809f-3e5254cefa05/dataset/data.yaml'
    cfg_yaml = '/opt/ml/ultralytics/ultralytics/cfg/models/v8/yolov8m_custom_z.yaml'
    # model = YOLO(cfg_yaml)
    # model.load(pretrained_model_path) # loading pretrain weights
    model = YOLO("/root/.cache/pretrain_models/3f9bcec16890299ad529d0cd0c3579a8.pt")
    lr = 0.01
    model.train(data=data_path,
                imgsz=640,
                lr0=lr,
                epochs=20,
                batch=16,
                close_mosaic=10,
                workers=1,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train_ms',
                name='exp',
                cache="disk"
                )


def train_yolo_v8():
    pretrained_model_path = "/root/.cache/ckpts/yolov8l.pt"
    data_path = '/opt/product/test_datas/brake_disc_test/dataset/data.yaml'
    model = YOLO(pretrained_model_path)
    lr = 0.01
    model.train(data=data_path,
                imgsz=1024,
                lr0=lr,
                epochs=100,
                batch=8,
                close_mosaic=10,
                workers=3,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train_disc',
                name='exp',
                # cache="disk"
                )
    
def train_yolo_v9():
    pretrained_model_path = "/root/.cache/ckpts/yolov9e.pt"
    data_path = '/opt/product/test_datas/brake_disc_test/dataset/data.yaml'
    model = YOLO(pretrained_model_path)
    lr = 0.01
    model.train(data=data_path,
                imgsz=640,
                lr0=lr,
                epochs=100,
                batch=16,
                close_mosaic=10,
                workers=3,
                device='1',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train_disc',
                name='exp',
                # cache="disk"
                )
    ...


if __name__ == '__main__':
    # train_yolo_v9()
    train_yolo_v8()