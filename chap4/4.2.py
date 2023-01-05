import cv2
import numpy as np

img = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
print('img.shape =', img.shape) #512x512

img = img.flatten() #262144, 1차원 배열로 변경
print('img.shape =', img.shape)

img = img.reshape(-1,512,512)
print('img.shape =', img.shape)

cv2.imshow('img',img[0]) #img[0]=512x512
cv2.waitKey()
cv2.destroyAllWindows()
