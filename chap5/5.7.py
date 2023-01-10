import cv2
import numpy as np


src = np.array([[0,0,0,0],
               [1,1,3,5],
               [6,1,1,3],
               [4,3,1,7]
               ], dtype=np.uint8)

hist = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[4], ranges=[0,8])
print('hist =', hist)

# [[9.] 0~1
#  [3.] 2~3
#  [2.] 4~5
#  [2.]]6~7

backP = cv2.calcBackProject([src], [0], hist, [0,8], scale=1) #src에서의 각 픽셀값을 해당하는 히스토그램 빈도수로 바꿈
print('backP =', backP)

# [[0,0,0,0],
# [1,1,3,5],
# [6,1,1,3],
# [4,3,1,7]]

# ~~back projection~~

# [[9 9 9 9]
#  [9 9 3 2]
#  [2 9 9 3]
#  [2 3 9 2]]
