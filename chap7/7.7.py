import cv2
import numpy as np

src = np.zeros(shape=(512,512,3), dtype=np.uint8)
cv2.rectangle(src, (50,100), (450,400), (255,255,255), -1)
cv2.rectangle(src, (100,150), (400,350), (0,0,0), -1)
cv2.rectangle(src, (200,200), (300,300), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

mode = cv2.RETR_LIST #모든 윤곽선을 찾는다
method = cv2.CHAIN_APPROX_SIMPLE #윤곽선의 다각형 근사 좌표를 변환한다
contours, hierarahy = cv2.findContours(gray, mode, method)

print('len(contours) =', len(contours)) #3
print('contours[0].shape =', contours[0].shape) #4,1,2
print('contours =', contours) 

for cnt in contours: #4,1,2 x 3
    cv2.drawContours(src, [cnt], 0, (255,0,0), 3)
    
    for pt in cnt: #4 pts
        cv2.circle(src, (pt[0][0], pt[0][1]), 5, (0,0,255), -1)
        
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()