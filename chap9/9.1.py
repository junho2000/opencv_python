import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/chessboard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

fastF = cv2.FastFeatureDetector.create(threshold=100) #임계값 지정해서 객체 생성
kp = fastF.detect(gray) #메소드로 특징점 검출
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255)) #특징점 표시
print('len(kp) =', len(kp))
cv2.imshow('dst', dst)

fastF.setNonmaxSuppression(False) #지역 극값 억제 X -> 특징점 개수 증가(비슷한곳에서 많이 나옴)
kp2 = fastF.detect(gray) 
dst2 = cv2.drawKeypoints(src, kp2, None, color=(0,0,255))
print('len(kp2) =', len(kp2))
cv2.imshow('dst2', dst2)

dst3 = src.copy()
points = cv2.KeyPoint_convert(kp) #직접 그릴 수 있게 좌표 변환
points = np.int32(points)

for cx, cy in points:
    cv2.circle(dst3, (cx, cy), 3, color=(255,0,0), thickness=1)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()
