import cv2
import numpy as np

#저대비 어두운 영상
src = np.array([[2,2,4,4],
                [2,2,4,4],
                [4,4,4,4],
                [4,4,4,4]],
               dtype=np.uint8)

#0~255로 히스토그램 평활화 -> 고대비 영상
dst = cv2.equalizeHist(src)
print('dst =', dst)

#히스토그램 평활화를 알고리즘으로 직접구현
hist, bins = np.histogram(src.flatten(), 256, [0,256])
cdf = hist.cumsum() #hist -> cdf transform
cdf_m = np.ma.masked_equal(cdf, 0) #0 마스킹
T = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) #cdf에서 0을 제외한 최소값과 최대값 계산
T = np.ma.filled(T,0).astype('uint8') #마스킹을 다시 0으로 채우기
dst2 = T[src]
print('dst2 =', dst2)