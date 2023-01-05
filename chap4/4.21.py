import cv2
import numpy as np
import time

dst = np.full((512,512,3), (255,255,255), dtype=np.uint8)
nPoints = 100
pts = np.zeros((1,nPoints,2), dtype=np.uint16) #0으로 초기화된 배열 1x100x2 생성, uint16 0 ~ 65535 ...??
cv2.setRNGSeed(int(time.time())) #난수 생성 초기화
cv2.randu(pts, (0,0), (512,512)) #pts배열에 (0,0)에서 (512,512)범위의 균등분포 난수를 생성

for k in range(nPoints):
    x, y = pts[0,k][:]
    cv2.circle(dst, (x,y), radius=5, color=(0,0,255), thickness=-1)
    
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()