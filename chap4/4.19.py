import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
src2 = np.zeros(shape=(512,512), dtype=np.uint8) + 255

dst1 = 255 - src1 #numpy의 broadcasting
dst2 = cv2.subtract(src2, src1)
dst3 = cv2.compare(dst1, dst2, cv2.CMP_NE) #dst1, dst2의 각 화소를 비교, not equal to, 참이면 255 거짓이면 0을 dst3의 각 화소에 출력
n    = cv2.countNonZero(dst3) #0이 아닌 화소를 카운트해서 반환 -> 화소 전부 0이기 때문에 0 반환

print('n =', n)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()