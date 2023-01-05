import cv2
import numpy as np

imageFile = '/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png'
img = cv2.imread(imageFile) #color
img2 = cv2.imread(imageFile, 0) #grayscale

cv2.imshow('lena color', img)
cv2.imshow('lena gray', img2)

cv2.waitKey()
cv2.destroyAllWindows()