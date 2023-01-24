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

res = cv2.preCornerDetect(gray, ksize=3) #Sobel 미분 연산자를 이용하여 계산
ret, res2 = cv2.threshold(np.abs(res), 1, 0, cv2.THRESH_TOZERO) #threshold 0.1보다 작은 값은 0으로 변경
corners = findLocalMaxima(res2)
print('corners.shape =', corners.shape)

res = cv2.normalize(np.abs(res), None, 0, 255, cv2.NORM_MINMAX) #코너점 주위에 여러개의 값이 생겨서 지역 극대값을 찾아야함
cv2.imshow('normalized |result| of preCornerDetect1', res)

dst = src.copy()
for x, y, in corners:
    cv2.circle(dst, (x,y), 5, (0,0,255), 1)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
