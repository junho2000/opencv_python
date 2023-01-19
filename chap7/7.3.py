import cv2
import numpy as np
img = np.zeros(shape=(512,512,3), dtype=np.uint8)
x1, x2 = 150, 350
y1, y2 = 150, 350
cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), -1)

# img = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/hough.jpg')

edges = cv2.Canny(img, 50, 100)
# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=250, minLineLength=200, maxLineGap=10) #rho : r 값의 범위 (0 ~ 1 실수), theta : 𝜃 값의 범위(0 ~ 180 정수), threshold : 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=100)
print('lines.shape =', lines.shape) #4, 1, 2 -> 4개의 직선의 양 끝점(x1,y1,x2,y2)

for line in lines:
    x1, y1, x2, y2 = line[0]
    print('x1,y1,x2,y2 =', x1, y1, x2, y2)
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)

cv2.imshow('edges', edges)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
