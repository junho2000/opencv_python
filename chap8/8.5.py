import cv2
import numpy as np


src = np.full((512,512,3), (0,0,0), np.uint8)
cv2.rectangle(src, (128,128), (384,384), (255,255,255), -1)
cv2.rectangle(src, (64,64), (256,256), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

res = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.01)
# 2x2 공분산 계산

res = cv2.dilate(res, None) #3x3 kernel
ret, res = cv2.threshold(res, 0.01*res.max(), 255, cv2.THRESH_BINARY)
res8 = np.uint8(res)
cv2.imshow('res8', res8)

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res8) #객체 검출
print('centroids.shape =', centroids.shape) #배경을 포함하기 떄문에 9x2
print('centroids =', centroids)
centroids = np.float32(centroids)
dst2 = src.copy()
for x, y in centroids: #배경 제외
    cv2.circle(dst2, (int(x), int(y)), 5, (0,0,255), 2)
cv2.imshow('dst2', dst2)

term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
corners = cv2.cornerSubPix(gray, centroids, (5,5), (-1,-1), term_crit)
print('corners =', corners)

corners = np.round(corners) #반올림
dst = src.copy()
for x, y in corners[1:]: #배경 제외
    cv2.circle(dst, (int(x), int(y)), 5, (0,0,255), 2)
    
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()