import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/chessboard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

fastF = cv2.FastFeatureDetector_create() #객체 fastF 생성, 기본 임계값은 10
kp = fastF.detect(gray)
dst = cv2.drawKeypoints(gray, kp, None, color=(255,0,0))
print('len(kp) =', len(kp))

kp = sorted(kp, key=lambda f: f.response, reverse=True) #특징점을 반응값 기준으로 내림차순 정렬
cv2.drawKeypoints(gray, kp[:10], dst, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG) #반응값 기준으로 큰 10개의 점을 파란색 원으로 표시, 결과 이미지 새로 생성 안 함
cv2.imshow('dst', dst)

kp2 = list(filter(lambda f: f.response > 50, kp))
print('len(kp2) =', len(kp2))

dst2 = cv2.drawKeypoints(gray, kp2, None, color=(0,0,255))
cv2.imshow('dst2', dst2)

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

kp3 = filteringByDistance(kp2, 30)
print('len(kp3) =', len(kp3))
dst3 = cv2.drawKeypoints(gray, kp3, None, color=(0,0,255))
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()
