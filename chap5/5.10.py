import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

nPoints = 100000
pts1 = np.zeros((nPoints, 1), dtype=np.uint16)
pts2 = np.zeros((nPoints, 1), dtype=np.uint16)

cv2.setRNGSeed(int(time.time())) #난수 초기화
cv2.randn(pts1, mean=(128), stddev=(10)) #난수 생성 평균 128, 표편 10
cv2.randn(pts2, mean=(110), stddev=(20)) #난수 생성 평균 110, 표편 20

H1 = cv2.calcHist(images=[pts1], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# cv2.normalize(H1, H1, 1, 0, cv2.NORM_L1) 

H2 = cv2.calcHist(images=[pts2], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# cv2.normalize(H2, H2, 1, 0, cv2.NORM_L1)

print(H1.shape, H2.shape)

S1 = np.zeros((H1.shape[0], 2), dtype=np.float32)
S2 = np.zeros((H1.shape[0], 2), dtype=np.float32)

for i in range(S1.shape[0]):
    S1[i,0] = H1[i,0]
    S2[i,0] = H2[i,0]
    S1[i,1] = i
    S2[i,1] = i
    #S1에 H1의 값 대입, 0~255까지 순서도 대입
    #S1,S2의 0열에 가중치로 H1,H2를 복사, 1열에 배열의 위치 정보 대입


#두 분포가 주어질 때, 하나의 분포를 다른 하나의 분포로 변경하는데 드는 최소 비용 계산(같은 분포는 EMD 0)
emd1, lowerBound, flow = cv2.EMD(S1, S2, cv2.DIST_L1) #cv2.DIST_L1 distance = |x1 - x2| + |y1 - y2|
print('EMD(S1, S2, DIST_L1) =', emd1)

emd2, lowerBound, flow = cv2.EMD(S1, S2, cv2.DIST_L2) #cv2.DIST_L2 distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
print('EMD(S1, S2, DIST_L2) =', emd2)

emd3, lowerBound, flow = cv2.EMD(S1, S2, cv2.DIST_C) #cv2.DIST_C distance = max(|x1 - x2|, |y1 - y2|)
print('EMD(S1, S2, DIST_C) =', emd3)

plt.plot(H1, color='r', label='H1')
plt.plot(H2, color='b', label='H2')
plt.legend(loc='best')
plt.show()