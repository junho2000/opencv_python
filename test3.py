import cv2
import numpy as np

#1
src = np.full((512,512,3), (255,255,255), np.uint8)
x, y = src.shape[:2]
x, y = x // 7, y // 5
print(x, y)
vx, vy = x, y
print(vx, vy)

for i in range(4):
    for j in range(6):  
        cv2.circle(src, (x, y), radius=10, color=(0,0,0), thickness=-1)
        x += vx
    x = vx
    y += vy
    
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
    