import cv2
import numpy as np

img = np.zeros(shape=(512,512,3), dtype=np.uint8)
cy = img.shape[0] // 2 #몫 연산자 -> 512/2 = 256
cx = img.shape[1] // 2

cv2.circle(img, (cx, cy), radius=25, color=(0,0,255), thickness=-1)
cv2.circle(img, (cx//2, cy), radius=65, color=(0,0,255), thickness=-1)
cv2.circle(img, (cx, cy//2), radius=55, color=(0,0,255), thickness=-1)
cv2.circle(img, (cx//2, cy//2), radius=85, color=(0,0,255), thickness=-1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#엣지가 아닌 1채널 8비트 그레이스케일 영상, method는 cv2.HOUGH_GRADIENT밖에 없음, dp=1이면 원본사이즈 dp=2이면 가로세로 반, 검출된 원들의 중심사이의 최소 거리, param1은 Canny의 threshold2이고 낮은 임계값은 param1/2이다, param2는 어큐뮬레이터의 임계값 
circles1 = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param2=15)

circles1 = np.int32(circles1)
print('circles1.shape =', circles1.shape)
for circle in circles1[0,:]: #원점(x,y), 반지름 r
    cx, cy, r = circle
    cv2.circle(img, (cx,cy), r, (0,255,0), 2)
cv2.imshow('img', img)

cv2.waitKey()
cv2.destroyAllWindows()