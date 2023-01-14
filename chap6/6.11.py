import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/T.jpg', cv2.IMREAD_GRAYSCALE)
ret, A = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY)
skel_dst = np.zeros(src.shape, np.uint8)
B = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
done = True
while done:
    erode = cv2.erode(A,B)
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, B)
    tmp = cv2.subtract(erode, opening)
    skel_dst = cv2.bitwise_or(skel_dst, tmp)
    A = erode.copy()
    done = cv2.countNonZero(A) != 0
    
cv2.imshow('src', src)
cv2.imshow('skel_dst', skel_dst)
cv2.waitKey()
cv2.destroyAllWindows()