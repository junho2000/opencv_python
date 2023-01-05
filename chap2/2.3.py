import cv2
from matplotlib import pyplot as plt

#opencv는 BGR로 읽어서 matplot사용할 땐 RGB로 바꿔야한다
imageFile = '/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png'
imgBGR = cv2.imread(imageFile)
plt.axis('off')

imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB)
plt.show()