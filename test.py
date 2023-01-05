import numpy as np
import cv2

src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/opencv_log.png')
cv2.imshow('img',src1)

print(src1.shape)


cv2.waitKey()
cv2.destroyAllWindows()
