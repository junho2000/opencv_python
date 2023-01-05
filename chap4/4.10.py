import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
shape = src.shape[0], src.shape[1], 3 #512x512x3
dst = np.zeros(shape, dtype=np.uint8) #3채널 컬러 영상 dst 생성

dst[:,:,0] = src #B <- Gray

dst[100:400, 200:300, :] = [255,255,255]

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()
