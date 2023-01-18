import cv2
import numpy as np

src = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
cy = src.shape[0] // 2 #몫 연산자 -> 512/2 = 256
cx = src.shape[1] // 2
cv2.circle(src, (cx, cy), radius=25, color=0, thickness=-1)
cv2.circle(src, (cx//2, cy), radius=65, color=0, thickness=-1)
cv2.circle(src, (cx, cy//2), radius=55, color=0, thickness=-1)
cv2.circle(src, (cx//2, cy//2), radius=85, color=0, thickness=-1)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, res = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res)
print('ret =',ret)
print('stats =', stats) 
print('centroids =', centroids)

dst = np.zeros(src.shape, dtype=src.dtype)
for i in range(1, int(ret)):
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)
    dst[labels == i] = [b,g,r]
    
for i in range(1, int(ret)):
    x, y, width, height, ares = stats[i]
    cv2.rectangle(dst, (x,y), (x+width, y+height), (0,0,255), 2)
    cx, cy = centroids[i]
    cv2.circle(dst, (int(cx), int(cy)), 5, (255,0,0), -1)
    
cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()