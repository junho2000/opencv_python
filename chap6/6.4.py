import cv2
import numpy as np
import matplotlib.pyplot as plt
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True) #그래디언트 크기와 각도 계산
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(angle)
print('angle :', minVal, maxVal, minLoc, maxLoc)
print('mag size :', mag.shape)
print('angle size :', angle.shape)

ret, edge = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY) #임계값 100으로 설정
edge = edge.astype(np.uint8) #화면표시를 위해 uint8로 변경
cv2.imshow('edge with threshold 100', edge)


height, width = mag.shape[:2] #512, 512
angleM = np.full((height, width, 3), (255,255,255), dtype=np.uint8) #mag와 같은 크기인 흰색 배경 생성
for y in range(height):
    for x in range(width):
        if edge[y,x] != 0: #threshold된 edge에 값이 있을 경우(그림 자세히보면 색깔별로 점 찍혀있음)
            if angle[y,x] == 0:
                angleM[y,x] = (0,0,255) #red
            elif angle[y,x] == 90:
                angleM[y,x] = (0,255,0) #green
            elif angle[y,x] == 180:
                angleM[y,x] = (255,0,0) #blue
            elif angle[y,x] == 270:
                angleM[y,x] = (0,255,255) #yellow
            else:
                angleM[y,x] = (128, 128, 128) #gray
cv2.imshow('angleM', angleM)

hist = cv2.calcHist(images=[angle], channels=[0], mask=edge, histSize=[360], ranges=[0,360]) #edge를 mask로 사용해 edge에서만 히스토그램을 구함.(검출된 엣지에서만 angle히스토그램을 구함)
hist = hist.flatten()
plt.plot(hist, color='r')
binX = np.arange(360)
plt.bar(binX, hist, width=1, color='b')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()