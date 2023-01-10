import cv2
import numpy as np


src = np.array([[0,0,0,0],
               [1,1,3,5],
               [6,1,1,3],
               [4,3,1,7]
               ], dtype=np.uint8)

#mask 빼고 전부 반드시 리스트로 입력, channel은 RGB면 3, 마스크 지정, x축 크기(bin)는 4, 0 ~ 7 
hist1 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[4], ranges=[0,8])
print('hist1 =', hist1)

# [[9.]   -> 0 ~ 1 count is 9
#  [3.]   -> 2 ~ 3 count is 3
#  [2.]   -> 4 ~ 5 count is 2
#  [2.]]  -> 6 ~ 7 count is 2

hist2 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[4], ranges=[0,4])
print('hist2 =', hist2)

# [[4.]   -> 0 count is 4
#  [5.]   -> 1 count is 5
#  [0.]   -> 2 count is 0
#  [3.]]  -> 3 count is 3