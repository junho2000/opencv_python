from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

#yolov8x.pt(최대) --> 잘 인식하긴 하는데 이걸로는 jetson nano로 실시간으로 처리하기는 무리일 수 있을 듯
#yolov8n.pt(최저) --> 대부분 인식이 안되거나 다른 것으로 인식함 --> 따로 데이터를 학습시키면 잘 인식 할지도? 속도가 그나마 제일 빠름
#yolov8s.pt(최저에서 2번째) --> 인식이 잘되는데 가끔 다른거나 인식을 못할 때가 있음 --> 최소한으로 실시간으로 처리 할수 있을 듯?

model = YOLO("yolov8s.pt")

model.predict(source = '/Users/kimjunho/Desktop/OpenCV_study/videos/circle_detect.mp4', save=False, conf=0.5, save_txt=False, show=True) #비디오 재생
# model.predict(source = '/Users/kimjunho/Desktop/OpenCV_study/bus.jpg', save=True, conf=0.5, save_txt='bus_yolo.jpg') #사진에 적용후 저장


# results = model.predict(source="0", show=True) #0번 카메라
# print(results)