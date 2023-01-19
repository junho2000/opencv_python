#예제7.11 cv2.distanceTransform(), cv2.watershed() 영상 분할
import cv2
import numpy as np

#1
src = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
cy = src.shape[0] // 2 #몫 연산자 -> 512/2 = 256
cx = src.shape[1] // 2
cv2.circle(src, (cx, cy), radius=25, color=0, thickness=-1)
cv2.circle(src, (cx//2, cy), radius=65, color=0, thickness=-1)
cv2.circle(src, (cx, cy//2), radius=55, color=0, thickness=-1)
cv2.circle(src, (cx//2, cy//2), radius=85, color=0, thickness=-1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(bImage, cv2.DIST_L1, 3)
dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX,
                      dtype = cv2.CV_8U)
cv2.imshow('bImage', bImage)
cv2.imshow('dist8', dist8)

#2 겹쳐진 원 분리를 위해 cv2.distanceTransform()사용
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
print('dist:', minVal, maxVal, minLoc, maxLoc)
mask = (dist > maxVal * 0.5).astype(np.uint8) * 255
cv2.imshow('mask', mask)

#3 
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(mask, mode, method)
print('len(contours)=', len(contours))

markers = np.zeros(shape=src.shape[:2], dtype = np.int32)
for i, cnt in enumerate(contours):
    cv2.drawContours(markers, [cnt], 0, i + 1, -1)

#4
dst = src.copy()
cv2.watershed(src, markers)

dst[markers == -1] = [0, 0, 255]        #경계선
for i in range(len(contours)):          #분할영역
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)
    dst[markers == i+1] = [b, g, r]
dst = cv2.addWeighted(src, 0.4, dst, 0.6, 0)    #합성

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()