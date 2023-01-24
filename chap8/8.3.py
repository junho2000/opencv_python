import cv2
import numpy as np


src = np.full((512,512,3), (0,0,0), np.uint8)
cv2.rectangle(src, (128,128), (384,384), (255,255,255), -1)
cv2.rectangle(src, (64,64), (256,256), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

eigen = cv2.cornerMinEigenVal(gray, blockSize=5) #가장 작은 고유값 계산
print('eigen.shape =', eigen.shape)

T = 0.2
corners = np.argwhere(eigen > T)
corners[:,[0,1]] = corners[:,[1,0]]
print('len(corners) =', len(corners))
dst = src.copy()

for x, y in corners:
    cv2.circle(dst, (x,y), 3, (0,0,255), 2)
    
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()