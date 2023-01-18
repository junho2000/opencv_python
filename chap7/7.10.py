import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/hand.jpg')
# src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/berlin.jpeg')
mask = np.zeros(shape=src.shape[:2], dtype=np.uint8) #윤곽선을 검출하기 위한 영상 
markers = np.zeros(shape=src.shape[:2], dtype=np.int32) #워터쉐드 분할을 위한 마커 영상
dst = src.copy()
cv2.imshow('dst',dst)

def onMouse(event, x, y, flags, param): #param[0] is mask, param[1] is dst
    if event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(param[0], (x,y), 10, (255,255,255), -1)
            cv2.circle(param[1], (x,y), 10, (255,255,255), -1)
    cv2.imshow('dst', param[1])

mode = cv2.RETR_EXTERNAL #가장 외곽의 윤곽선을 찾음
method = cv2.CHAIN_APPROX_SIMPLE #다각형의 근사 좌표를 변환
while True:
    cv2.setMouseCallback('dst', onMouse, [mask, dst])
    key = cv2.waitKey(30)
    
    if key == 0x1B: #ESC
        break
    elif key == ord('r'): #r키를 누르면 리셋
        mask[:,:] = 0 
        dst = src.copy()
        cv2.imshow('dst', dst)
    elif key == ord(' '): #스페이스바를 누르면 영역 분할
        contours, hierarchy = cv2.findContours(mask, mode, method)
        print('len(contours) =', len(contours))
        markers[:,:] = 0
        for i, cnt in enumerate(contours):
            cv2.drawContours(markers, [cnt], 0, i+1, -1) #-1 윤곽선 내부를 채움
        cv2.watershed(src, markers)
        
        dst = src.copy()
        dst[markers == -1] = [0,0,255] #경계선을 빨간색으로 변경
        for i in range(len(contours)):
            r = np.random.randint(256)    
            g = np.random.randint(256)
            b = np.random.randint(256)
            dst[markers == i+1] = [b,g,r]
        
        dst = cv2.addWeighted(src, 0.4, dst, 0.6, 0) #srcx0.4 + dstx0.6의 가중치로 화면에 띄움
        cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()