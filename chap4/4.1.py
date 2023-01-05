import cv2
import numpy as np

img = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_COLOR)
#img = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE) 
# img.ndim = 2
# img.shape = (512, 512)
# img.dtype = uint8
# img.dtype = int32
# img.dtype = uint32

cv2.imshow('img', img) #uint8자료형의 영상만을 화면에 표시

#original pic attribute
print('img.ndim =', img.ndim) #3-dimension BGR
print('img.shape =', img.shape) #512x512x3
print('img.dtype =', img.dtype) #uint8 0 ~ 255

#change attribute
img = img.astype(np.int32)
print('img.dtype =', img.dtype)
img = np.uint32(img)
print('img.dtype =', img.dtype)

cv2.waitKey()
cv2.destroyAllWindows()