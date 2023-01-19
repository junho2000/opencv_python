import cv2
import numpy as np

# src = np.zeros(shape=(512,512,3), dtype=np.uint8)
# cv2.rectangle(src, (50,100), (450,400), (255,255,255), -1)
# cv2.rectangle(src, (100,150), (400,350), (0,0,0), -1)
# cv2.rectangle(src, (200,200), (300,300), (255,255,255), -1)

src = np.zeros(shape=(512,512,3), dtype=np.uint8)
cy = src.shape[0] // 2 
cx = src.shape[1] // 2
cv2.circle(src, (cx, cy), radius=25, color=(0,255,255), thickness=-1)
cv2.circle(src, (cx//2, cy), radius=65, color=(0,255,0), thickness=-1)
cv2.circle(src, (cx, cy//2), radius=55, color=(0,0,255), thickness=-1)
cv2.circle(src, (cx//2, cy//2), radius=85, color=(255,255,0), thickness=-1)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

mode1 = cv2.RETR_EXTERNAL #가장 외곽의 윤곽선만 찾는다 -> len(contours) = 1, contours[0].shape = (4, 1, 2)
mode2 = cv2.RETR_LIST #모든 윤곽선을 찾는다 -> len(contours) = 3, contours[0].shape = (4, 1, 2)
method1 = cv2.CHAIN_APPROX_SIMPLE #윤곽선의 다각형 근사 좌표를 변환한다 -> 다각형의 꼭짓점만 변환(원도 가능)
method2 = cv2.CHAIN_APPROX_NONE #윤곽선의 모든 좌표를 변환한다
contours, hierarchy = cv2.findContours(gray, mode1, method1) #배경이 검은 1채널 8비트 이미지, 모드, 방법, contours(검출된 윤곽선), hierarchy(윤곽선의 계증구조)
print('type(contours) =', type(contours))
print('type(contours[0]) =', type(contours[0]))
print('len(contours) =', len(contours))
print('conours[0].shape =', contours[0].shape)
print('contours[0] =', contours[0])

cv2.drawContours(src, contours, -1, (255,0,0), 3)

# for pt in contours[0][:]: #첫번째 윤곽선
#     cv2.circle(src, (pt[0][0], pt[0][1]), 5, (0,0,255), -1)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()