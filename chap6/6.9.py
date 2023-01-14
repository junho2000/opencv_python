import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/Morphology_Closing.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/Morphology_Opening.png', cv2.IMREAD_GRAYSCALE)

kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3)) #모폴로지 연산을 위한 여러 모양의 커널을 생성하는 함수, 3x3의 사각형 커널 생성

erode = cv2.erode(src2, kernel, iterations=1) #주변을 1칸씩 깍음 -> 침식(erode), 1번 반복
dilate = cv2.dilate(src1, kernel, iterations=2) #주변을 1칸씩 채움 -> 확장(dilate), 1번 반복
erode2 = cv2.erode(dilate, kernel, iterations=2)

cv2.imshow('src1', src1)
cv2.imshow('src2', src2)
cv2.imshow('src2 with erode', erode)
cv2.imshow('src1 with dilate', dilate)
cv2.imshow('dilate and erode', erode2)
cv2.waitKey()
cv2.destroyAllWindows()

