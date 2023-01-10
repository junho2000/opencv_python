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
cv2.normalize(H1, H1, 1, 0, cv2.NORM_L1) #정규화 이전의 데이터, 정규화 이후의 데이터, 정규화구간1,2, flag 전체 합으로 나누는 cv2.NORM_L1 -> sum(H1)은 1
plt.plot(H1, color='r', label='H1')

H2 = cv2.calcHist(images=[pts2], channels=[0], mask=None, histSize=[256], ranges=[0,256])
cv2.normalize(H2, H2, 1, 0, cv2.NORM_L1)

#통계적 방법에 기초해서 히스토그램 분포 비교 
d1 = cv2.compareHist(H1, H2, cv2.HISTCMP_CORREL) #Correlation Coefficient 계산 |d1|<=1
d2 = cv2.compareHist(H1, H2, cv2.HISTCMP_CHISQR) #d2가 작을 수록 유사한 히스토그램
d3 = cv2.compareHist(H1, H2, cv2.HISTCMP_INTERSECT) #d3가 클수록 유사한 히스토그램 (정규화된 히스토그램에서는 0<=d3<=1)
d4 = cv2.compareHist(H1, H2, cv2.HISTCMP_BHATTACHARYYA) #d4값이 작을 수록 유사한 히스토그램 (정규화된 히스토그램에서만 사용가능 0<=d4<=1)
print('d1(H1, H2, CORREL) =', d1)
print('d2(H1, H2, CHISQR) =', d2)
print('d3(H1, H2, INTERSECT) =', d3)
print('d4(H1, H2, BHATTACHARYYA) =', d4)

plt.plot(H2, color='b', label='H2')
plt.legend(loc='best')
plt.show()

