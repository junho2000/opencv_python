import cv2
import numpy as np

#1
src = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/chessboard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
patternSize = (7, 7) #내부 코너점 8x8

#체스보드 패턴의 내부 코너점을 순차적으로 검출하고 그리는 함수
#시작점은 왼쪽 위 또는 오른쪽 아래점을 기준으로 행우선 순서로 검출하여 corners에 반환
found, corners = cv2.findChessboardCorners(src, patternSize) #그레이스케일 또는 컬러 영상, 패턴의 내부 코너점의 열과 행의 크기
print('corners.shape=', corners.shape) #7x7 = 49개의 코너점의 좌표(x,y)

#2
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term_crit) #부화소 수준으로 계산

#3
dst = src.copy()
cv2.drawChessboardCorners(dst, patternSize, corners2, found) #found=True

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()