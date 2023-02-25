import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Inference
model('/Users/kimjunho/Desktop/OpenCV_study/pictures/lenna.png')

# Results


