import cv2
from matplotlib import pyplot as plt

imageFile = '/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png'
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.imshow(imgGray, cmap='gray')

plt.axis('off')
plt.savefig('/Users/kimjunho/Desktop/python_workspace/pictures/lenna_plt.png')
plt.show()