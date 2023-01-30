from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("yolov8x.pt")

# model.predict(source = '/Users/kimjunho/Desktop/OpenCV_study/videos/vtest.mp4', save=False, conf=0.5, save_txt=False, show=True) #비디오 재생
# model.predict(source = '/Users/kimjunho/Desktop/OpenCV_study/bus.jpg', save=True, conf=0.5, save_txt='bus_yolo.jpg') #사진에 적용후 저장
results = model.predict(source="0", show=True) #0번 카메라

print(results)