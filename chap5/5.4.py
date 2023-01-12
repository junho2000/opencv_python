import cv2
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
hist1 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[32], ranges=[0,256])

hist2 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[256], ranges=[0,256])

print(hist1.shape, hist2.shape) #(32, 1) (256, 1)
hist1 = hist1.flatten()
hist2 = hist2.flatten()
print(hist1.shape, hist2.shape) #(32,) (256,)

plt.title('hist1: binX =  np.arange(32)')
plt.plot(hist1, color = 'r') #y 값이 32개가 있어서 자동으로 x값은 0~31로 지정
binX = np.arange(32)
plt.bar(binX, hist1, width=1, color='b') #x value, y value, 막대폭 지정(default is 0.8) 1로 지정하면 막대가 서로 붙음, color blue
plt.show()

plt.title('hist1: binX =  np.arange(32) * 8')
binX = np.arange(32) * 8 #256
plt.plot(binX, hist1, color='r') #8칸 간격으로 plot 생성, x, y, color
plt.bar(binX, hist1, width=8, color='b')
plt.show()

plt.title('hist2: binX =  np.arange(256)')
plt.plot(hist2, color = 'r') 
binX = np.arange(256)
plt.bar(binX, hist2, width=1, color='b')
plt.show()