import cv2
import numpy as np

src = np.full((512,512,3), (255,255,255), dtype=np.uint8)
cv2.rectangle(src, (50,50), (200,200), (0,0,255), 2)
cv2.circle(src, (300,300), 100, (0,0,255), 2)

dst = src.copy()
cv2.floodFill(dst, mask=None, seedPoint=(100,100), newVal=(255,0,0)) #(100,100)을 시작점으로 사각형 내부를 newVal로 채운다

retval, dst2, mask, rect = cv2.floodFill(dst, mask=None, seedPoint=(300,300), newVal=(0,255,0)) #seedPoint를 시작으로 원 내부를 newVal로 채운다
print('retval =', retval)
print('mask.shape =', mask.shape) #512+2, 512+2, 채워진 영역을 1로 채움
mask = cv2.normalize(mask,None,0,255,cv2.NORM_MINMAX)
print('rect =', rect)
x, y, width, height = rect #채워진 영역의 바운딩 사각형을 반환한다
cv2.rectangle(dst2, (x,y), (x+width, y+height), (255,0,0), 2)

cv2.imshow('mask', mask)
cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()