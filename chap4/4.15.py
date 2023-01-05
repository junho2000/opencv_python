import cv2
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')

dst1 = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE) #시계방향 90도 회전
dst2 = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE) #반시계방향 90도 회전

cv2.imshow('dst1',dst1)
cv2.imshow('dst2',dst2)
cv2.waitKey()
cv2.destroyAllWindows()
