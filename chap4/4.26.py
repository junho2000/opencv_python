import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')
b,g,r = cv2.split(src)

cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)

X = src.reshape(-1,3) #flatten -> 262144x3
print('X.shape =', X.shape)

mean, eVects = cv2.PCACompute(X, mean=None) #X의 평균 벡터, 공분산 행렬의 고유벡터를 계산
print('mean =', mean) # 105.41025  99.051216  180.22366
print('eVects =', eVects)
# [[ 0.39645132  0.6897987   0.60580885]
#  [-0.6426597  -0.26271227  0.71970195]
#  [ 0.6556028  -0.67465556  0.33915314]]

Y = cv2.PCAProject(X, mean, eVects) #수직한 고유벡터들로 투영
Y = Y.reshape(src.shape)
print('Y.shape =', Y.shape) #512x512x3


eImage = list(cv2.split(Y))
for i in range(3): #Y의 각 채널을 normalize
    cv2.normalize(eImage[i], eImage[i], 0, 255, cv2.NORM_MINMAX)
    eImage[i] = eImage[i].astype(np.uint8)

#eImage[0]은 고유값이 가장 큰 고유벡터, 그다음은 eImage[1]...
cv2.imshow('eImage[0]', eImage[0])
cv2.imshow('eImage[1]', eImage[1])
cv2.imshow('eImage[2]', eImage[2])
cv2.waitKey()
cv2.destroyAllWindows()