import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.medianBlur(src, ksize=7)
dst2 = cv2.blur(src, ksize=(7,7))
dst3 = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=0.0)
dst4 = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=10.0)

cv2.imshow('dst1 median filter', dst1)
cv2.imshow('dst2 blur', dst2)
cv2.imshow('dst3 Gaussian when sigma is 0.0', dst3)
cv2.imshow('dst4 Gaussian when sigma is 10.0', dst4)
cv2.waitKey()
cv2.destroyAllWindows()