import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
dst = np.zeros(src.shape, dtype=src.dtype)

N = 32
height, width = src.shape

h = height // N
w = width // N
#h, w는 각 블록의 크기
for i in range(N):
    for j in range(N):
        y = i * h
        x = j * w
        #x, y는 각 블록의 왼쪽 상단
        roi = src[y:y + h, x:x + w]
        dst[y:y + h, x:x + w] = cv2.mean(roi)[0] #gray
        #cv2.mean()은 각가의 채널을 독립적으로 평균값을 계산한다
        #dst[y:y + h, x:x + w] = cv2.mean(roi)[0:3] #컬러 영상
        
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
        
    