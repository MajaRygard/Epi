import torch

# Load YOLOv5 model (YOLOv5 is available via PyTorch Hub)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is a small version

def detect_objects(image_path):
    # Run object detection
    results = model(image_path)
    return results
