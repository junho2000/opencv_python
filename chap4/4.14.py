import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
dst = cv2.resize(src, dsize=(320, 240)) #가로, 세로
dst2 = cv2.resize(src, dsize=(0,0), fx = 1.5, fy = 1.2)

print(dst.shape) #240, 320
print(dst2.shape) # 614, 768

cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()