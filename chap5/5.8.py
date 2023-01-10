import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

roi = cv2.selectROI(src)
print('roi', roi)
roi_h = h[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] #h의 ROI 영역
hist = cv2.calcHist([roi_h], [0], None, [64], [0,256]) #ROI 영역을 64빈 히스토그램으로 계산
print(hist)
backP = cv2.calcBackProject([h.astype(np.float32)], [0], hist, [0,256], scale=1.0) #h 를 hist로 역투영한 backP 계산
print(backP)

hist = cv2.sort(hist, cv2.SORT_EVERY_COLUMN+cv2.SORT_DESCENDING) #열을 내림차순으로 정렬
print(hist)
k = 1
T = hist[k][0] - 1 #hist에서 k + 1번째까지 가장 많은 분포의 화소를 검출하기 위한 임계값 설정
print('T =', T)
ret, dst = cv2.threshold(backP, T, 255, cv2.THRESH_BINARY) #k + 1번째로 많은 화소들의 

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()