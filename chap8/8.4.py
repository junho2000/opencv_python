import cv2
import numpy as np

def findLocalMaxima(src): #팽창과 침식의 모폴로지 연산으로 지역 극대값의 좌표를 point배열에 검출하여 반환
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11,11))
    
    dilate = cv2.dilate(src,kernel) #이웃의 최대값 계산
    localMax = (src == dilate)
    
    erode = cv2.erode(src,kernel) #이웃의 최소값 계산
    localMax2 = src > erode
    localMax &= localMax2
    points = np.argwhere(localMax == True) #행, 열 순서로 찾음
    points[:,[0,1]] = points[:,[1,0]] #열, 행 순서로 바꿈
    return points

src = np.full((512,512,3), (0,0,0), np.uint8)
cv2.rectangle(src, (128,128), (384,384), (255,255,255), -1)
cv2.rectangle(src, (64,64), (256,256), (255,255,255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

res = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.01)
ret, res = cv2.threshold(np.abs(res), 0.02, 0, cv2.THRESH_TOZERO) #less than threshold 0.2 -> 0
res8 = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('res8', res8)

corners = findLocalMaxima(res)
print('corners =', corners)

corners = corners.astype(np.float32, order= 'C') #C언어 스타일의 메모리 구조 지정
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.01) #max_iter 인자에 지정된 횟수만큼 반복하고 중단, 주어진 정확도(epsilon 인자)에 도달하면 반복을 중단
corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term_crit) #코너점의 위치를 부화소 수준으로 다시 계산
print('corners2 =', corners2) #corners랑 다름

dst = src.copy()

for x, y in np.int32(corners2):
    cv2.circle(dst, (x,y), 3, (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()