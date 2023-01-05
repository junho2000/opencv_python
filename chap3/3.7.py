import cv2
import numpy as np

img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255

x, y = 256, 256
size = 200

for angle in range(0,90,10): #0 ~ 80
  rect = ((256,256), (size,size), angle) #중심, 양변의 길이, 각도도
  box = cv2.boxPoints(rect).astype(np.int32) #사각형의 모서리점을 계산
  r = np.random.randint(256) # 0~256중 랜덤
  g = np.random.randint(256)
  b = np.random.randint(256)
  cv2.polylines(img, [box], True, (b,g,r), 2) #img, 다각형들의 numpy배열, 닫힘, 색깔, 두께

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()