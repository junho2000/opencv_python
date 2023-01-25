import cv2
import numpy as np

roi = None
drag_start = None
mouse_status = 0
tracking_start = False

def onMouse(event, x, y, flags, param=None):
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
            mouse_status = 2
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_status = 3
        
cv2.namedWindow('tracking')
cv2.setMouseCallback('tracking', onMouse)

cap = cv2.VideoCapture(0) ##
if (not cap.isOpened()):
    print('Error opening Video')
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
roi_mask = np.zeros((height, width), dtype=np.uint8)
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1)

t = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    t += 1
    print('t =', t)

    frame2 = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0., 60., 32.), (180., 255., 255.))
    
    if mouse_status == 2:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        
    if mouse_status == 3:
        print('initialize.....')
        mouse_status = 0
        x1, y1, x2, y2 = roi
        mask_roi = mask[y1:y2, x1:x2]
        hsv_roi = hsv[y1:y2, x1:x2]
        
        hist_roi = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0,180])
        cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)
        track_window1 = (x1, y1, x2 - x1, y2 - y1)
        track_window2 = (x1, y1, x2 - x1, y2 - y1)
        tracking_start = True
    
    if tracking_start:
        backP = cv2.calcBackProject([hsv], [0], hist_roi, [0,180], 1)
        backP &= mask
        cv2.imshow('backP', backP)
        
        ret, track_window1 = cv2.meanShift(backP, track_window1, term_crit)
        x, y, w, h = track_window1
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        
        track_box, track_window2 = cv2.CamShift(backP, track_window2, term_crit)
        x, y, w, h = track_window2
        cv2.rectangle(frame2, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.ellipse(frame2, track_box, (0,255,255), 2)
        pts = cv2.boxPoints(track_box)
        pts = np.int0(pts)
        dst = cv2.polylines(frame2, [pts], True, (0,0,255), 2)
        
        cv2.imshow('tracking', frame)
        cv2.imshow('CamShift tracking', frame2)
        
        key = cv2.waitKey(25)
        if key is 27:
            break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows() 