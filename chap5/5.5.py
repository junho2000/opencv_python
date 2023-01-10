import cv2
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')
histColor = ('b', 'g', 'r')
for i in range(3):
    hist = cv2.calcHist(images=[src], channels=[i], mask=None, histSize=[256], ranges=[0,256])
    #src, channel 012, no mask, histbin scope 0~255, input range 0~255 
    plt.plot(hist, color=histColor[i])
plt.show()