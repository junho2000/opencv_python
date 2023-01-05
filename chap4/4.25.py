import cv2
import numpy as np

X = np.array([[0,0,0,100,100,150,-100,-150],[0,50,-50,0,30,100,-20,-100]],dtype=np.float64)
X = X.transpose()

mean,eVects = cv2.PCACompute(X,mean=None)
print('mean =',mean)
print('eVects =',eVects)

Y = cv2.PCAProject(X,mean,eVects) #PCA 투영
print('Y =',Y)

X2 = cv2.PCABackProject(Y,mean,eVects) #PCA 역투영
print('X2 =',X2)
print(np.allclose(X,X2)) #PCA 역투영 함으로서 원본 복구 X = X2 is True
  
cv2.waitKey()
cv2.destroyAllWindows()