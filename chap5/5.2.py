import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src',src)

#전체 이미지에 하나의 threshold 적용
ret, dst = cv2.threshold(src, 0, 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU) #2개의 값을 갖는 이진 영상 생성
cv2.imshow('dst',dst) 

#화소마다 다른 임계값을 적용하는 적응형 임계값 영상을 계산
dst2 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 7) #각 픽셀 주변 51x51의 mean - 7(c)가 threshold
cv2.imshow('dst2',dst2)

dst3 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 7)
cv2.imshow('dst3',dst3)

cv2.waitKey()
cv2.destroyAllWindows()