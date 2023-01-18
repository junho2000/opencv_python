import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')

down2 = cv2.pyrDown(src) #입력영상을 가우시안 필터링하고 가로 세로 1/2배로 축소한 피라미드 영상
down4 = cv2.pyrDown(down2)
print('down2.shape =', down2.shape)
print('down4.shape =', down4.shape)

up2 = cv2.pyrUp(src) #입력영상을 가우시안 필터링하고 가로 세로 2배로 축소한 피라미드 영상
up4 = cv2.pyrUp(up2)
print('up2.shape =', up2.shape)
print('up4.shape =', up4.shape)

cv2.imshow('down2', down2)
cv2.imshow('up2', up2)
cv2.waitKey()
cv2.destroyAllWindows()