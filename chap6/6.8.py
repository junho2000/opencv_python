import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

def logFilter(ksize = 7): #가우시안 함수에 대한 2차 미분에의한 라플라시안을 계산하여 윈도우 필터를 생성(수학적으로는 잘 모르겠음)
    k2 = ksize // 2
    sigma = 0.3 * (k2 - 1) + 0.8 #필터의 크기
    print('sigma =', sigma)
    LoG = np.zeros((ksize,ksize), dtype=np.float32)
    for y in range(-k2, k2 + 1):
        for x in range(-k2, k2 + 1):
            g = -(x * x + y * y) / (2.0 * sigma ** 2.0)
            LoG[y + k2, x + k2] = -(1.0 + g) * np.exp(g) / (np.pi * sigma ** 4.0)
    return LoG

kernel = logFilter()
LoG = cv2.filter2D(src, cv2.CV_32F, kernel)
cv2.imshow('LoG', LoG)

def zeroCrossing2(lap, thresh=0.01):
    height, width = lap.shape
    Z = np.zeros(lap.shape, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighbors = [lap[y-1,x], lap[y+1,x],
                         lap[y,x-1], lap[y,x+1],
                         lap[y-1,x-1],lap[y-1,x+1],
                         lap[y+1,x-1],lap[y+1,x+1]]
            pos = 0
            neg = 0
            for value in neighbors:
                if value > thresh:
                    pos += 1
                if value < -thresh:
                    neg += 1
            if pos > 0 and neg > 0:
                Z[y,x] = 255    
    return Z
edgeZ = zeroCrossing2(LoG)
cv2.imshow('LoG with Zero Crossing2', edgeZ)
cv2.waitKey()
cv2.destroyAllWindows()