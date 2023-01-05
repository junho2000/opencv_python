import cv2
import numpy as np

img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
cy = img.shape[0] // 2 #몫 연산자 -> 512/2 = 256
cx = img.shape[1] // 2

for r in range(200, 0, -100): #200, 100
  cv2.circle(img, (cx, cy), r, color = (255,0,0)) #그림img, 중심점, 반지름, 색

cv2.circle(img, (cx, cy), radius=50, color=(0,0,255), thickness=-1)

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()