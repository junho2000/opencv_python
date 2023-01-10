import cv2
import numpy as np

X = np.array([[0,0,0,100,100,150,-100,-150],
              [0,50,-50,0,30,100,-20,-100]],
              dtype=np.float64)
X = X.transpose()

cov, mean = cv2.calcCovarMatrix(X, mean=None, flags=cv2.COVAR_NORMAL + cv2.COVAR_ROWS)
print('mean =', mean) #mean = [[12.5   1.25]]
print('cov =', cov)
# [[73750.  34875. ]
#  [34875.  26287.5]]

ret , icov = cv2.invert(cov)
print('icov =', icov)
# [[ 3.63872307e-05 -4.82740722e-05]
#  [-4.82740722e-05  1.02084955e-04]]

v1 = np.array([[0],[0]], dtype=np.float64)
v2 = np.array([[0],[50]], dtype=np.float64)

dist = cv2.Mahalanobis(v1,v2,icov)
print('dist =', dist)
#dist = 0.5051854992128457

cv2.waitKey()
cv2.destroyAllWindows()


