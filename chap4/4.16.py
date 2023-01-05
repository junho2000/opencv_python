import cv2
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')

rows, cols, channels = src.shape
M1 = cv2.getRotationMatrix2D((rows/2, cols/2), 45, 0.5) #변환행렬 M1 2x3
M2 = cv2.getRotationMatrix2D((rows/2, cols/2), -45, 1.0) #변환행렬 M2 2x3

dst1 = cv2.warpAffine(src, M1, (rows, cols))  #rows, cols 크기의 영상 생성
dst2 = cv2.warpAffine(src, M2, (rows, cols))

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()