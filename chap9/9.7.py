import cv2
import numpy as np

def distance(f1, f2): #Euclidean distance
    x1, y1 = f1.pt
    x2, y2 = f2.pt
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def filteringByDistance(kp, distE=0.5): #반응값 기준으로 내림차순 정렬된 특징점들 중에서 거리오차보다 작은 점들을 삭제
    size = len(kp)
    mask = np.arange(1, size+1).astype(np.bool_) #1 ~ size만큼 True
    for i, f1 in enumerate(kp):
        if not mask[i]: #False면 넘어감
            continue
        else:
            for j, f2 in enumerate(kp):
                if i == j: #같은 점이면 넘어감
                    continue
                if distance(f1, f2) < distE: #두 점 사이의 거리가 작으면 False
                    mask[j] = False
    np_kp = np.array(kp)
    return list(np_kp[mask])

src = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/chessboard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0.0)

fastF = cv2.FastFeatureDetector_create(threshold=30)
mserF = cv2.MSER_create(10)
blobF = cv2.SimpleBlobDetector_create()
goodF = cv2.GFTTDetector_create(maxCorners=20, minDistance=10)

kp = fastF.detect(gray)
# kp = mserF.detect(gray)
# kp = blobF.detect(gray)
# kp = goodF.detect(gray)
print('len(kp) =', len(kp))

filtered_kp = filteringByDistance(kp, 10) #거리 10 이하 특징점 삭제
print('len(filtered_kp) =', len(filtered_kp))
dst = cv2.drawKeypoints(gray, filtered_kp, None, color=(0,0,255))
cv2.imshow('dst', dst)

orbF = cv2.ORB_create()
filtered_kp, des = orbF.compute(gray, filtered_kp) #디스크립터 계산
print('des.shape =', des.shape)
print('des =', des)
dst2 = cv2.drawKeypoints(gray, filtered_kp, None, color=(0,0,255))

for f in filtered_kp:
    x, y = f.pt
    size = f.size
    rect = ((x,y), (size,size), f.angle) #회전 사각형 정의
    box = cv2.boxPoints(rect).astype(np.int32) #모서리 좌표 읽기
    cv2.polylines(dst2, [box], True, (0,255,0), 2)
    cv2.circle(dst2, (round(x), round(y)), round(f.size / 2), (255,0,0), 2)

cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()