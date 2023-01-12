import numpy as np
import cv2

src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
print(src1.size)
hist1 = cv2.calcHist(images=[src1], channels=[0], mask=None, histSize=[256], ranges=[0,256])
print('hist1 =', hist1)