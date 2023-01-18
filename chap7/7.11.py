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
ret, bImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(bImage, cv2.DIST_L1, 3) #색깔이 없는 곳과 있는 곳까지의 거리 계산
dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #정규화
cv2.imshow('bImage', bImage)
cv2.imshow('dist8', dist8)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
print('dist :', minVal, maxVal, minLoc, maxLoc)
mask = (dist > maxVal * 0.5).astype(np.uint8) * 255 #최대값의 절반보다 크면 마스크
cv2.imshow('mask', mask)

mode = cv2.RETR_EXTERNAL #바깥쪽의 경계선만 검출
method = cv2.CHAIN_APPROX_SIMPLE #다각형의 꼭짓점들 반환
contours, hierarchy = cv2.findContours(mask, mode, method)
print('len(contours) =', len(contours))

markers = np.zeros(shape=img.shape[:2], dtype=np.int32)
for i, cnt in enumerate(contours):
    cv2.drawContours(markers, [cnt], 0, i+1, -1)
    
dst = img.copy()
cv2.watershed(img, markers)

dst[markers == -1] = [0,0,255] #경계선 빨간색 지정
for i in range(len(contours)): #랜덤 색깔 지정
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)
    dst[markers == i+1] = [b,g,r]
dst = cv2.addWeighted(img, 0.4, dst, 0.6, 0) #원본영상과 가중치둬서 출력

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()