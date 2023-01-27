import cv2
import numpy as np

#1
src = np.full((512,512,3), (255,255,255), np.uint8)
x, y = src.shape[:2]
x, y = x // 7, y // 5
vx, vy = x, y
for i in range(4):
    for j in range(6):  
        cv2.circle(src, (x, y), radius=10, color=(0,0,0), thickness=-1)
        x += vx
    x = vx
    y += vy

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

patternSize = (6, 4)
found, corners = cv2.findCirclesGrid(src, patternSize)
print('corners.shape=', corners.shape) #6x4 = 24개의 원 검출, (x,y)

#2
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term_crit) #부화소 계산

#3
dst = src.copy()
cv2.drawChessboardCorners(dst, patternSize, corners2, found)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()