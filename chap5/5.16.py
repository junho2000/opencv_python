import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

#영상 전체에 하나의 히스토그램을 이용하여 dst에 평활화
dst = cv2.equalizeHist(src)
cv2.imshow('dst equalized', dst)

#하나의 히스토그램을 사용하기 때문에 dst와 비슷함, 하지만 Clahe는 히스토그램 재분배를 하고 테이블 계산 방법이 다르기 때문에 정확히 같지는 않다
clahe2 = cv2.createCLAHE(clipLimit=40, tileGridSize=(1,1))
dst2 = clahe2.apply(src)
cv2.imshow('dst2 CLAHE with (1,1)', dst2)

#8x8개의 타일로 나누어 각 히스토그램 계산, 히스토그램 재분배, 경계선 보간 -> 전체적으로 대비가 선명한 영상을 얻는다
clahe3 = cv2.createCLAHE(clipLimit=40, tileGridSize=(8,8))
dst3 = clahe3.apply(src)
cv2.imshow('dst3 CLAHE with (8,8)', dst3)

cv2.waitKey()
cv2.destroyAllWindows()