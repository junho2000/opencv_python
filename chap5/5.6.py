import cv2
import matplotlib.pyplot as plt
import numpy as np

#2d histogram을 만들 수 있다는 건 알겠는데 히스토그램에서 각 픽셀의 의미를 잘 모르겠음.
bgr = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')
hist01 = cv2.calcHist([bgr], [0,1], None, [32,32], [0,256,0,256])
print(hist01.size) #1024 = 32 x 32

plt.title('hist01')
plt.ylim(0,31)
plt.imshow(hist01, interpolation="nearest") #interpolation은 보간법을 뜻하며, 픽셀들의 축 위치 간격을 보정하여 이미지가 자연스러운 모양으로 보일 수 있게 하는 방법
plt.show()

hist02 = cv2.calcHist([bgr], [0,2], None, [32,32], [0,256,0,256])
plt.title('hist02')
plt.ylim(0,31)
plt.imshow(hist02, interpolation="nearest")
plt.show()

hist12 = cv2.calcHist([bgr], [1,2], None, [32,32], [0,256,0,256])
plt.title('hist12')
plt.ylim(0,31)
plt.imshow(hist12, interpolation="nearest")
plt.show()

