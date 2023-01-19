import cv2
import numpy as np

#pyrMeanShiftFiltering : Gaussian blur에 비해 속도는 10~20배정도 느리지만, filtering 성능은 매우 뛰어나다.
def floodFillPostProcess(src, diff = (2,2,2)):
    img = src.copy()
    rows, cols = img.shape[:2]
    mask = np.zeros(shape = (rows + 2, cols + 2), dtype = np.uint8)
    for y in range(rows):
        for x in range(cols):
            if mask[y+1, x+1] == 0: #배경 둘레는 1로 채워져 있음(안쪽만) 픽셀하나하나 floodFill 채워져있으면 x
                r = np.random.randint(256)
                g = np.random.randint(256)
                b = np.random.randint(256)
                cv2.floodFill(img, mask, (x,y), (b,g,r), diff, diff)
    return img


src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/hand.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
dst = floodFillPostProcess(src)
dst2 = floodFillPostProcess(hsv)
cv2.imshow('src', src)
cv2.imshow('hsv', hsv)
cv2.imshow('floodFillProcess with src', dst)
cv2.imshow('floodFillProcess with hsv', dst2)

res = cv2.pyrMeanShiftFiltering(src, sp=5, sr=20, maxLevel=4)
dst3 = floodFillPostProcess(res)

term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2)
res2 = cv2.pyrMeanShiftFiltering(hsv, sp=5, sr=20, maxLevel=4, termcrit=term_crit)
dst4 = floodFillPostProcess(res2)

cv2.imshow('pyrMeanShift with src', res)
cv2.imshow('pyrMeanShift with hsv', res2)
cv2.imshow('floodFillProcess with pyrShifted src', dst3)
cv2.imshow('floodFillProcess with pyrShifted hsv', dst4)
cv2.waitKey()
cv2.destroyAllWindows()