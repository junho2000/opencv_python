import cv2
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) #BGR -> GARY
yCrCv = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb) #BGR -> YCrCv
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) #BGR -> HSV

cv2.imshow('gray', gray)
cv2.imshow('yCrCv', yCrCv)
cv2.imshow('hsv', hsv)

cv2.waitKey()
cv2.destroyAllWindows()