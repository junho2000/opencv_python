import cv2

imageFile = '/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png'
img = cv2.imread(imageFile)

cv2.imwrite('/Users/kimjunho/Desktop/python_workspace/pictures/lenna1.png', img)
cv2.imwrite('/Users/kimjunho/Desktop/python_workspace/pictures/lenna2.png', img, [cv2.IMWRITE_PNG_COMPRESSION,9])
cv2.imwrite('/Users/kimjunho/Desktop/python_workspace/pictures/lenna3.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])