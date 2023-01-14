import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

kx, ky = cv2.getDerivKernels(1,0,ksize=3) #x축 방향 미분필터 kx ky 생성
sobelX = ky.dot(kx.T) #kx를 전치하고 ky와 내적해서 ksize=3에서의 1차 미분필터를 만듬
print('kx =', kx)
print('ky =', ky)
print('sobelX =', sobelX)
gx = cv2.filter2D(src, cv2.CV_32F, sobelX) #x축 소벨필터를 적용해 gx 생성

kx, ky = cv2.getDerivKernels(0,1,ksize=3) #y축 방향 미분필터 kx ky 생성
sobelY = ky.dot(kx.T) #ksize=3에서의 1차 미분필터를 만듬
print('kx =', kx)
print('ky =', ky)
print('sobelY =', sobelY)
gy = cv2.filter2D(src, cv2.CV_32F, sobelY) #y축 소벨필터를 적용해 gy 생성

mag = cv2.magnitude(gx, gy) #x축 y축 그래디언트로 이루어진 그래디언트의 크기를 구함
ret, edge = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY) #임계값 100으로 적용

cv2.imshow('edge', edge)
cv2.waitKey()
cv2.destroyAllWindows()
