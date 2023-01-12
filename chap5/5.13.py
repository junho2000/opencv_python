#컬러영상의 평활화 적용 -> 대비 증가
import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')
cv2.imshow('src', src)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv) #색상(Hue), 채도(Saturation), 명도(Value)

v2 = cv2.equalizeHist(v) #히스토그램 평활화로 대비를 증가시키고 선명하게 하고 싶은면 HSV에서 V를 평활화 적용
hsv2 = cv2.merge([h,s,v2])
dst = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
cv2.imshow('dst HSV', dst)

yCrCv = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
y, Cr, Cv = cv2.split(yCrCv) #Y는 휘도(색의 밝고 어두움) 성분이며 Cb 와 Cr 은 색차 성분

y2 = cv2.equalizeHist(y) #히스토그램 평활화로 대비를 증가시키고 선명하게 하고 싶은면 yCrCv에서 y를 평활화 적용
yCrCv = cv2.merge([y2, Cr, Cv])
dst2 = cv2.cvtColor(yCrCv, cv2.COLOR_YCrCb2BGR)

cv2.imshow('dst2 YCrCb', dst2)
cv2.waitKey()
cv2.destroyAllWindows()

