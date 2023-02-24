import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Inference
model('/Users/kimjunho/Desktop/OpenCV_study/videos/vtest.mp4')

# Results


