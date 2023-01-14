import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3) #dx=1, dy=0이면 x축편미분, 그래디언트를 계산
gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3) #dx=0, dy=1이면 y축편미분
print('gx, gy size :',gx.shape, gy.shape)

dstX = cv2.sqrt(np.abs(gx)) #그래디언트의 크기를 구함
dstX = cv2.normalize(dstX, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #그래디언트를 0~255 사이즈로 정규화

dstY = cv2.sqrt(np.abs(gy))
dstY = cv2.normalize(dstY, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

mag = cv2.magnitude(gx, gy) #소벨 필터로 구한 x방향, y방향 미분 값을 cv2.magnitude에 입력값으로 설정하면 벡터의 크기를 계산
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mag)
print('mag :', minVal, maxVal, minLoc, maxLoc)
print('mag size', mag.shape)
dstM = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #정규화

cv2.imshow('src', src)
cv2.imshow('dstX', dstX)
cv2.imshow('dstY', dstY)
cv2.imshow('dstM', dstM)
cv2.waitKey()
cv2.destroyAllWindows()