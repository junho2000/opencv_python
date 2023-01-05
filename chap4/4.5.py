import cv2
import numpy as np

img = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')

#512x512x(B,G,R)
img[100:400, 200:300, 0] = 255 #B
img[100:400, 300:400, 1] = 255 #G
img[100:400, 400:500, 2] = 255 #R

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
