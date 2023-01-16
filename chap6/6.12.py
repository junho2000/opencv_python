import cv2
import numpy as np
#템플릿 매칭이란 reference image에서 template과의 매칭 위치를 탐색하는 방법
#물체인식, 스테레오 영상에서 대응점 검출에 사용될 수 있다
#translation(이동) problem is solvable, but not for roatation and scaling problem.

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/alphabet.png', cv2.IMREAD_GRAYSCALE)
tmp_A = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/A.png', cv2.IMREAD_GRAYSCALE)
tmp_S = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/S.png', cv2.IMREAD_GRAYSCALE)
tmp_b = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/b.png', cv2.IMREAD_GRAYSCALE)
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

R1 = cv2.matchTemplate(src, tmp_A, cv2.TM_SQDIFF_NORMED)
minVal, _, minLoc, _ = cv2.minMaxLoc(R1)
print('TM_SQDIFF_NORMED :', minVal, minLoc)
h, w = tmp_A.shape[:2]
cv2.rectangle(dst, minLoc, (minLoc[0]+w, minLoc[1]+h), (255,0,0), 2)

R2 = cv2.matchTemplate(src, tmp_S, cv2.TM_CCORR_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R2)
print('TM_CCORR_NORMED :', maxVal, maxLoc)
h, w = tmp_S.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0]+w, maxLoc[1]+h), (0,255,0), 2)

R3 = cv2.matchTemplate(src, tmp_b, cv2.TM_CCOEFF_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R3)
print('TM_CCOEFF_NORMED :', maxVal, maxLoc)
h, w = tmp_b.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0]+w, maxLoc[1]+h), (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()