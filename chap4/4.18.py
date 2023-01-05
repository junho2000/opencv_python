import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')
src2 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/opencv_log.png') #예제와는 달리 배경이 검은색인 logo
src2 = cv2.resize(src2, (src2.shape[0] // 5, src2.shape[1] // 5)) #logo_size = 900,963
cv2.imshow('src2', src2)

rows, cols, channels = src2.shape
roi = src1[0:rows, 0:cols] #src2의 전체크기에 대한 src1의 영역을 roi에 저장

gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) #src, 임계값, 최대값
mask_inv = cv2.bitwise_not(mask)

temp = mask_inv
mask_inv = mask
mask = temp

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)

src1_bg = cv2.bitwise_and(roi, roi, mask = mask) #roi roi and연산 -> roi,그리고 mask 와 and 연산
cv2.imshow('src1_bg', src1_bg)

src2_fg = cv2.bitwise_and(src2, src2, mask = mask_inv)
cv2.imshow('src2_fg', src2_fg)

dst = cv2.bitwise_or(src1_bg, src2_fg) #lena배경인 roi에 logo의 바깥영역 + logo 안영역
cv2.imshow('dst', dst)

src1[0:rows, 0:cols] = dst

cv2.imshow('result', src1)
cv2.waitKey(0)
cv2.destroyAllWindows()




