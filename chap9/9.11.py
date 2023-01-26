import cv2
import numpy as np

#SIFT는 이미지의 Scale (크기) 및 Rotation (회전)에 Robust한 (= 영향을 받지 않는) 특징점을 추출하는 알고리즘이고 계산량이 많고 정확도가 높다.
#하지만 오래걸려서 임베디드에는 적합하지 않음, 임베디드에 사용한다면 ORB가 좋을듯


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

siftF = cv2.SIFT_create(edgeThreshold=80) #SIFT 객체 생성
kp = siftF.detect(gray)
print('len(kp) =', len(kp))

kp = sorted(kp, key=lambda f: f.response, reverse=True)
# filtered_kp = list(filter(lambda f: f.response > 0.01, kp))
filtered_kp = filteringByDistance(kp, 10)
print('len(filtered_kp) =', len(filtered_kp))

kp, des = siftF.compute(gray, filtered_kp) #디스크립터 des 계산
print('des.shape =', des.shape)
print('des.dtype =', des.dtype)
print('des =', des)

dst = cv2.drawKeypoints(src, filtered_kp, None, color=(0,0,255))

for f in filtered_kp: #show scale, gradient angle in SIFT
    x, y = f.pt
    size = f.size
    rect = ((x,y), (size, size), f.angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(dst, [box], True, (0,255,0), 2)
    cv2.circle(dst, (round(x), round(y)), round(f.size / 2), (255,0,0), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()