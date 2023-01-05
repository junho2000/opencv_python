import cv2
import numpy as np

X = np.array([[0,0,0,100,100,150,-100,-150],[0,50,-50,0,30,100,-20,-100]],dtype=np.float64)
X = X.transpose()

cov,mean = cv2.calcCovarMatrix(X,mean=None, flags = cv2.COVAR_NORMAL+cv2.COVAR_ROWS) # 공분산 평균 구하기
mean = mean.transpose()

ret,icov = cv2.invert(cov) # 공분산 Transpose


dst = np.full((512,512,3),(255,255,255),dtype=np.float64) # 결과 행렬 초기화 및 Parameter 선언
rows,cols,channel = dst.shape

centerX = cols//2
centerY = rows//2

# EigenVecor, EigenVector를 활용하여 대칭인 점을 좌표를 계산하기 위하여 y축 대칭
v2 = np.zeros((2,1),dtype=np.float64)
FLIP_Y = lambda y:rows-1-y

#Mahalanobis distance를 활용하여 거리에 따른 색상 지정
for y in range(rows):
    for x in range(cols):
        v2[0,0] = x-centerX
        v2[1,0]=FLIP_Y(y) - centerY #Y축 뒤집기
        
        icov = np.array(icov,dtype=np.float64)
        dist = cv2.Mahalanobis(mean,v2,icov)
        
        if dist<0.1:
            dst[y,x] = [50,50,50]
        elif dist<0.3:
            dst[y,x] = [100,100,100]
        elif dist<0.8:
            dst[y,x] = [200,200,200]
        else:
            dst[y,x] = [250,250,250]

# X안의 x, y의 좌표값을 원으로서 표시하는 것
for k in range(X.shape[0]):
    x,y = X[k,:]
    cx = int(x+centerX)
    cy = int(y+centerY)
    cy = FLIP_Y(cy)
    cv2.circle(dst,(cx,cy),radius=5,color=(0,0,255),thickness=-1)
    
#draw X,Y-axes
cv2.line(dst,(0,256),(cols-1,256),(0,0,0))
cv2.line(dst,(256,0),(256,rows),(0,0,0))
    
# calculate eigen vectors
ret,eVals,eVects = cv2.eigen(cov)
print('eVals=',eVals)
print('eVects=',eVects)
    
def ptsEigenVector(eVal,eVect):
    scale = np.sqrt(eVal)
    x1 = scale * eVect[0]
    y1 = scale * eVect[1]
    x2,y2 = -x1,-y1
        
    x1 +=mean[0,0] + centerX
    y1 +=mean[1,0] + centerY
    x2 +=mean[0,0] + centerX
    y2 +=mean[1,0] + centerY
    y1=FLIP_Y(y1)
    y2=FLIP_Y(y2)
    return x1,y1,x2,y2
    
    
#draw eVects[0]
#EigenValue, EigenVector를 활용하여 데이터 X에 대한 분포의의 단축(short axis)그리기
x1,y1,x2,y2 = ptsEigenVector(eVals[0],eVects[0])
cv2.line(dst,(x1,y1),(x2,y2),(255,0,0),2)
    
#draw eVects[1]
#EigenValue, EigenVector를 활용하여 데이터 Y에 대한 분포의의 단축(short axis)그리기
x1,y1,x2,y2 = ptsEigenVector(eVals[1],eVects[1])
cv2.line(dst,(x1,y1),(x2,y2),(255,0,0),2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()