import cv2

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)

dst = src.copy()
#dst = src #하면 dst가 src를 참조하게됨->dst를 변경하면 src도 변경됨
dst[100:400, 200:300] = 0

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()
