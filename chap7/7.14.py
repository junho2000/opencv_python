import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/hand.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

data = src.reshape((-1,3)).astype(np.float32) #클러스터링을 위한 데이터(행) 한줄로 쭉핌
K = 2 #클러스터의 개수(군집 수)
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #종료조건(각 클러스터의 중심이 오차 이내로 움직이면 종료, 최대 반복횟수) 최대 10번 반복하고 1픽셀 이하로 움직이면 종료 
ret, labels, centers = cv2.kmeans(data, K, None, term_crit, 5, cv2.KMEANS_RANDOM_CENTERS) #클러스터의 중심을 초기화하는 방법(랜덤)
print('centers.shape =', centers.shape) #K개의 클러스터 중심점을 반환 k개의 (b,g,r) -> 2x3
print('labels.shape', labels.shape) #각 데이터 점의 클러스터 번호 반환
print('ret', ret) #클러스터링 밀집도 반환

centers = np.uint8(centers)
print(labels.flatten().shape)
res = centers[labels.flatten()]
dst = res.reshape(src.shape)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
