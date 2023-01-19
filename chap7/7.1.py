import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

edges1 = cv2.Canny(src, 50, 100) #threshol1, threshold2 -> hysteresis thresholding method
edges2 = cv2.Canny(src, 50, 200)

cv2.imshow('edges1 th1:50, th2:100', edges1)
cv2.imshow('edges2 th1:50, th2:200', edges2)
cv2.waitKey()
cv2.destroyAllWindows()
