import cv2
import numpy as np
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src',src)

ret, dst = cv2.threshold(src, 200, 255 , cv2.THRESH_BINARY) #thresh=200, max_val=255
print('ret =',ret) #ret = 200
cv2.imshow('dst',dst) 

ret2, dst2 = cv2.threshold(src, 200, 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu 알고리즘으로 최적의 임계값을 계산 이때 threshold값이 필요없음.
print('ret =',ret2) #ret = 124
cv2.imshow('dst2',dst2)

cv2.waitKey()
cv2.destroyAllWindows()