import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.medianBlur(src, ksize=7) #중위값을 사용함, salt and pepper noise의 경우엔 평균/가우시안 필터보다 효과적
dst2 = cv2.blur(src, ksize=(7,7)) #커널 내부의 합계를 계산하고 커널크기로 정규화시킴
dst3 = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=0.0) #위치에 따라 가우시안 가중치 적용, sigmaX,Y=0이면 커널크기로 계산
dst4 = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=10.0) #sigmaX,Y=10

cv2.imshow('dst1 median filter', dst1)
cv2.imshow('dst2 blur', dst2)
cv2.imshow('dst3 Gaussian when sigma is 0.0', dst3)
cv2.imshow('dst4 Gaussian when sigma is 10.0', dst4)
cv2.waitKey()
cv2.destroyAllWindows()