import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.boxFilter(src, ddepth=-1, ksize=(11,11)) #kernal size는 홀수로 지정, ksize가 커질수록 더 흐려짐
dst2 = cv2.boxFilter(src, ddepth=-1, ksize=(21,21))


#가우시안함수를 사용하여 가장자리(Edge)를 선명하게 보존하면서 노이즈를 우수하게 제거하는 흐림 효과 함수
dst3 = cv2.bilateralFilter(src, d=11, sigmaColor=10, sigmaSpace=10)
dst4 = cv2.bilateralFilter(src, d=-1, sigmaColor=10, sigmaSpace=10) #d=-1로 이웃의 크기가 sigmaSpace에 의해 결정

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.imshow('dst4', dst4)
cv2.waitKey()
cv2.destroyAllWindows()