import cv2
import numpy as np

src = np.full((512,512,3), (0,0,0), np.uint8)
cv2.rectangle(src, (128,128), (384,384), (255,255,255), -1)
cv2.rectangle(src, (64,64), (256,256), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

res = cv2.cornerEigenValsAndVecs(gray, blockSize=5, ksize=3)
# λ1, λ2 가 모두 작은 곳은 평평한 영역에 있는 점
# λ1, λ2 둘 중 하나는 크고 하나는 작으면 에지
# λ1, λ2 두 값이 모드 큰 경우 코너점

print('res.shape =', res.shape) #512,512,6
eigen = cv2.split(res) # -> λ1, λ2, (x1, y1), (x2, y2)
print('res =', res)

T = 0.2
ret, edge = cv2.threshold(eigen[0], T, 255, cv2.THRESH_BINARY) #큰 고유값
edge = edge.astype(np.uint8)

corners = np.argwhere(eigen[1] > T) #작은 고유값
corners[:,[0,1]] = corners[:,[1,0]] #switch x, y
print('len(corners) =', len(corners))

dst = src.copy()
for x, y in corners:
    cv2.circle(dst, (x,y), 5, (0,0,255), 2)

cv2.imshow('edge', edge)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
