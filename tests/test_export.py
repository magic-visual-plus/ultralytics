from ultralytics import YOLO

model = YOLO("/root/.cache/ckpts/yolov8n.pt")
# success = model.export(format="rknn", opset=12)
success = model.export(format="onnx", opset=12)
