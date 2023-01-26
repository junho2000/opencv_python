#예제 10.7 cv2.meanShift(), cv2.Camshift() 추적
import cv2
import numpy as np

#1 마우스 이벤트를 통해 roi 영역 설정
roi = None
drag_start = None
mouse_status = 0
tracking_start = False
def onMouse(event, x, y, flags, param = None):
    global roi
    global drag_start
    global mouse_status
    global tracking_start
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        mouse_status = 1
        tracking_start = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            xmin = min(x, drag_start[0])
            ymin = min(y, drag_start[1])
            xmax = max(x, drag_start[0])
            ymax = max(y, drag_start[1])
            roi = (xmin, ymin, xmax, ymax)
            mouse_status = 2    #dragging
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_status = 3        #complete

#2
cv2.namedWindow('tracking') #윈도우 생성
cv2.setMouseCallback('tracking', onMouse) #onMouse를 마우스 이벤트 콜백 함수로 설정

cap = cv2.VideoCapture(0)
if (not cap.isOpened()):
    print('Error opening video')
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) #프레임 크기 읽기
roi_mask = np.zeros((height, width), dtype= np.uint8) #roi_mask 0으로 초기화
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,    #종료조건 설정
             10, 1)

#3
t = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    t += 1
    print('t=', t)
#3-1
    frame2 = frame.copy()   #CamShift 추적 결과 표시를 위한 복사 영상
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0., 60., 32.), (180., 255., 255.)) #어둡고 흐릿한 영역 제외 #H[0, 180], S[60, 255], V[32, 255] 범위 포함 255, 아닌 영역은 0인 이진영상
##      cv2.imshow('mask', mask)
#3-2 마우스 드래깅시에 frame에 파란 사각형 표시
    if mouse_status==2:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
#3-3 
    if mouse_status == 3:
        print('initialize....')
        mouse_status = 0
        x1, y1, x2, y2 = roi
        mask_roi = mask[y1:y2, x1:x2]
        hsv_roi = hsv[y1:y2, x1:x2]         #roi 영역 저장

        hist_roi = cv2.calcHist([hsv_roi], [0], mask_roi,   #hsv_roi의 H채널 히스토그램 계싼
                                [16], [0, 180])
        cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)  #히스토그램 빈도수 정규화
        track_window1 = (x1, y1, x2 - x1, y2 - y1)  #meanshift
        track_window2 = (x1, y1, x2 - x1, y2 - y1)  #camshift
        tracking_start = True   #추적 시작
#3-4
    if tracking_start:
        backP = cv2.calcBackProject([hsv], [0], hist_roi, [0, 180], 1) #H채널 히스토그램 역투영한 backP생성
        backP &= mask   #AND연산 -> 역투영에서 어둡고 흐릿한 영역 제외
        cv2.imshow('backP', backP)

#3-5: meanShift tracking -> 역투영을 이용해서 관심영역 물체 추적
        ret, track_window1 = cv2.meanShift(backP, track_window1,
                                            term_crit)
        x, y, w, h = track_window1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

#3-6: camShift tracking -> 역투영을 통해 관심영역 물체 추적
        track_box, track_window2 = cv2.CamShift(backP,
                                                track_window2,
                                                term_crit)
        x, y, w, h = track_window2
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)   #초록색 사각형
        cv2.ellipse(frame2, track_box, (0, 255, 255), 2)    #노란색 타원
        pts = cv2.boxPoints(track_box)
        pts = np.int0(pts)      #np.int32
        dst = cv2.polylines(frame2,[pts],True, (0, 0, 255), 2)  #빨간색 회전 사각형
##          cv2.imshow('tracking', frame)
#           cv2.imshow('CamShift tracking', frame2)
#           cv2.waitKey(0)
    cv2.imshow('tracking', frame)                 # meanShift        
    cv2.imshow('CamShift tracking', frame2)       # CamShift
    key = cv2.waitKey(25)
    if key == 27:
        break
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()