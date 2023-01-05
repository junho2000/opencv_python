import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
src2 = np.zeros(shape=(512,512), dtype=np.uint8,) + 100

dst1 = src1 + src2 #넘파이 배열 덧셈을 해서 255를 넘으면 256으로 나눈 나머지를 계산
dst2 = cv2.add(src1, src2, dtype=cv2.CV_8U) #덧셈 결과가 255를 넘으면 255으로 계산

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()