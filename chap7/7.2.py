import cv2
import numpy as np

img = np.zeros(shape=(512,512,3), dtype=np.uint8)
x1, x2 = 150, 350
y1, y2 = 150, 350
cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), -1)

# img = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/hough.jpg')

edges = cv2.Canny(img, 50, 100)
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180.0, threshold=300) 
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180.0, threshold=100)
# rho : r 값의 범위 (0 ~ 1 실수), theta : 𝜃 값의 범위(0 ~ 180 정수), threshold : 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
print('lines.shape =', lines.shape) #4, 1, 2 -> 4개의 직선의 r, theta

for line in lines:
    rho, theta = line[0]
    c = np.cos(theta)
    s = np.sin(theta)
    x0 = c * rho #x0, y0는 원점에서 직선과 직각으로 만나는 좌표
    y0 = s * rho
    print('x0, y0, c, s =', x0, y0, c, s)
    x1 = int(x0 + 500 * (-s)) #단위벡터를 통해 스케일링
    y1 = int(y0 + 500 * (c))
    x2 = int(x0 - 500 * (-s))
    y2 = int(y0 - 500 * (c))
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1) #시작점, 끝점, 색, 두께

cv2.imshow('rec', img)
cv2.imshow('edges', edges)
cv2.waitKey()
cv2.destroyAllWindows()