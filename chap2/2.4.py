import cv2
from matplotlib import pyplot as plt

imageFile = '/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png'
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
plt.axis('off')

plt.imshow(imgGray, cmap='gray', interpolation='bicubic') #interpolation 보간법
plt.show()