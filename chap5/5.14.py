import cv2
import numpy as np

src = np.array([[2,2,2,2,0,0,0,0],
                [2,1,1,2,0,0,0,0],
                [2,1,1,2,0,0,0,0],
                [2,2,2,2,0,0,0,0],
                [0,0,0,0,255,255,255,255],
                [0,0,0,0,255,1,1,255],
                [0,0,0,0,255,1,1,255],
                [0,0,0,0,255,255,255,255]
                ], dtype=np.uint8)



clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(1,1)) 
#임계값 40, 타일그리드 크기 1,1 -> tileArea = 8*8 전체히스토그램 1개 계산
#clipLimit = 40 * 64 / 256 = 10
dst = clahe.apply(src)
print('dst=\n', dst)

clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(2,2)) 
#임계값 40, 타일그리드 크기 2,2 -> tileArea = 4*4, 히스토그램 4개 계산
#clipLimit = 40 * 16 / 256 = 2.5
dst2 = clahe.apply(src)
print('dst2=\n', dst2)