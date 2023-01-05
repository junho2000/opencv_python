import cv2
import numpy as np

img = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
img[100,200] = 0 #100행 200열 화소를 0으로 변경
print(img[100:110, 200:210]) 

img[100:400, 200:300] = 0 #100~400행 200~300열 화소를 0으로 변경

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
