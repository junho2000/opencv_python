import cv2
import numpy as np


src = np.full((512,512,3), (0,0,0), np.uint8)
cv2.rectangle(src, (128,128), (384,384), (255,255,255), -1)
cv2.rectangle(src, (64,64), (256,256), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

K = 5 #10
corners = cv2.goodFeaturesToTrack(gray, maxCorners=K, qualityLevel=0.05, minDistance=10)
print('corners.shape =', corners.shape)
print('corners =', corners)

corners2 = cv2.goodFeaturesToTrack(gray, maxCorners=K, qualityLevel=0.05, minDistance=10, useHarrisDetector=True, k=0.04)
print('corners2.shape =', corners2.shape)
print('corners2 =', corners2)

dst = src.copy()
corners = corners.reshape(-1,2)
for x, y in corners:
    cv2.circle(dst, (int(x), int(y)), 5, (0,0,255), -1)
    
corners2 = corners2.reshape(-1,2)
for x, y in corners2:
    cv2.circle(dst, (int(x), int(y)), 5, (255,0,0), 2)
    
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
