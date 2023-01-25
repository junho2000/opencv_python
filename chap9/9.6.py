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

src = np.full((512,512,3), (0,0,0), np.uint8)
cv2.rectangle(src, (128,128), (384,384), (255,255,255), -1)
cv2.rectangle(src, (64,64), (256,256), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0.0)

orbF = cv2.ORB_create(scoreType=1) #FAST_SCORE
kp = orbF.detect(gray)
print('len(kp) =', len(kp))

dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
cv2.imshow('dst', dst)

kp = sorted(kp, key=lambda f: f.response, reverse=True) #반응값 순으로 오름차순 정렬
filtered_kp = list(filter(lambda f: f.response > 50, kp)) #반응값이 50보다 넘는 것들 필터링
filtered_kp = filteringByDistance(kp, 10) #유클리디안 거리가 10이하인 점들 삭제
print('len(filterd_kp) =', len(filtered_kp))

kp, des = orbF.compute(gray, filtered_kp) #ORB 디스크립터 계산
print('des.shape =', des.shape) #8x32 -> 디스크립터는 8개의 특징점 각각에 대하여 32바이트이다
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