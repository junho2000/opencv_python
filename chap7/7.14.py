import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/hand.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

data = src.reshape((-1,3)).astype(np.float32)
K = 2
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, labels, centers = cv2.kmeans(data, K, None, term_crit, 5, cv2.KMEANS_RANDOM_CENTERS)
print('centers.shape =', centers.shape)
print('labels.shape', labels.shape)
print('ret', ret)

centers = np.uint8(centers)
res = centers[labels.flatten()]
dst = res.reshape(src.shape)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
