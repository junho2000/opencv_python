import cv2
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')

dst = cv2.split(src)
print(type(dst)) #tuple
print(type(dst[0])) #ndarray

cv2.imshow('blue', dst[0])
cv2.imshow('green', dst[1])
cv2.imshow('red', dst[2])
cv2.waitKey()
cv2.destroyAllWindows()