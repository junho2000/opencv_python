import cv2
import numpy as np

src0 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/Morphology_BlackHat.png', cv2.IMREAD_GRAYSCALE)
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/Morphology_TopHat.png', cv2.IMREAD_GRAYSCALE)
src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/Morphology_Closing.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/Morphology_Opening.png', cv2.IMREAD_GRAYSCALE)

kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))

closing = cv2.morphologyEx(src1, cv2.MORPH_CLOSE, kernel, iterations=1) #흰색 물체 속의 검은색 잡음을 제거
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1) #흰색 잡음을 제거
gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel, iterations=1) #흰색 물체의 테두리를 검출
tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel, iterations=1) #검은색 배경 속의 흰색 점을 검출 
blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel, iterations=1) #흰색 배경 속에 검은색 점을 검출

cv2.imshow('opening', opening)
cv2.imshow('closing', closing)
cv2.imshow('gradient', gradient)
cv2.imshow('tophat', tophat)
cv2.imshow('blackhat', blackhat)
cv2.waitKey()
cv2.destroyAllWindows()