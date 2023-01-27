import cv2
import numpy as np

src = np.full((512,512,3), (0,0,0), np.uint8)
cv2.rectangle(src, (128,128), (384,384), (255,255,255), -1)
cv2.rectangle(src, (64,64), (256,256), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

#영상 모멘트 계산
M = cv2.moments(bImage, True) #1채널 영상 또는 경계선 좌표 배열, True->화소값을 1로 처리
for key, value in M.items():
    print('{} = {}'.format(key, value))
    
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
dst = src.copy()
cv2.circle(dst, (cx, cy), 5, (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
