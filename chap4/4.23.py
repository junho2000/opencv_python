import cv2
import numpy as np

X = np.array([[0,0,0,100,100,150,-100,-150],
              [0,50,-50,0,30,100,-20,-100]],
              dtype=np.float64)
X = X.transpose()

cov, mean = cv2.calcCovarMatrix(X, mean=None, flags=cv2.COVAR_NORMAL + cv2.COVAR_ROWS)
print('mean =', mean)
print('cov =', cov)

ret , icov = cv2.invert(cov)
print('icov =', icov)

v1 = np.array([[0],[0]], dtype=np.float64)
v2 = np.array([[0],[50]], dtype=np.float64)

dist = cv2.Mahalanobis(v1,v2,icov)
print('dist =', dist)

cv2.waitKey()
cv2.destroyAllWindows()


