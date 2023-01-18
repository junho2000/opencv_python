import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/hand.jpg')

# Why using HSV color space for this?
# In real world images there is always variations in the image color values due to various lightening conditions, 
# shadows and, even due to noise added by the camera while clicking and subsequently processing the image.
# To over the above color variations, we will perform color detection in the HSV color space.

hsv1 = cv2.cvtColor(src1, cv2.COLOR_BGR2HSV)
lowerb1 = (0,15,0)
upperb1 = (20,180,255)
dst1 = cv2.inRange(hsv1, lowerb1, upperb1) #이진 영상

src2 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/berlin.jpeg')
hsv2 = cv2.cvtColor(src2, cv2.COLOR_BGR2HSV)
lowerb2 = np.array([0, 100, 20])
upperb2 = np.array([10, 255, 255]) 
dst2 = cv2.inRange(hsv2, lowerb2, upperb2) #이진 영상

cv2.imshow('src1', src1)
cv2.imshow('dst1', dst1)
cv2.imshow('src2', src2)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()