import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=0.0) #가우시안 필터로 노이즈 제거후
lap = cv2.Laplacian(blur, cv2.CV_32F, 3) #라플라시안 필터링
cv2.imshow('lap', lap)

ret, edge = cv2.threshold(np.abs(lap), 10, 255, cv2.THRESH_BINARY)
edge = edge.astype(np.uint8)
cv2.imshow('lap with Threshold', edge)

def SGN(x): #부호 확인 양수 -> +1, 음수 -> -1
    if x >= 0:
        sign = 1
    else:
        sign = -1
    return sign

def zeroCrossing(lap):
    height, width = lap.shape
    Z = np.zeros(lap.shape, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighbors = [lap[y-1,x], lap[y+1,x],
                         lap[y,x-1], lap[y,x+1],
                         lap[y-1,x-1],lap[y-1,x+1],
                         lap[y+1,x-1],lap[y+1,x+1]]
            mValue = min(neighbors)
            if SGN(lap[y,x]) != SGN(mValue): #특정 위치 주변의 제일 낮은 값의 부호가 반대일 경우 EDGE!
                Z[y,x] = 255
    return Z
edgeZ = zeroCrossing(lap)
cv2.imshow('Zero Crossing', edgeZ)
cv2.waitKey()
cv2.destroyAllWindows()