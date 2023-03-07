import cv2

trackers = [cv2.TrackerMIL_create,
            cv2.TrackerKCF_create,
            cv2.TrackerGOTURN_create, 
            cv2.TrackerCSRT_create]

trackerIdx = 0  
tracker = None
isFirst = True

video_src = 0 
video_src = "/Users/kimjunho/Desktop/OpenCV_study/videos/circle_detect.mp4"

cap = cv2.VideoCapture(video_src)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
win_name = 'Tracking APIs'

while cap.isOpened():
    ret, frame = cap.read()
    
    # (+추가) 동영상 크기 조정을 위한 코드
    frame=cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    
    if not ret:
        print('Cannot read video file')
        break
    
    img_draw = frame.copy()
    
    if tracker is None: 
        cv2.putText(img_draw, "Press the Space to set ROI!!", \
            (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    else:
        ok, bbox = tracker.update(frame)   
        (x,y,w,h) = bbox
        if ok: # 추적 성공
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), \
                          (0,255,0), 2, 1)
        else : # 추적 실패
            cv2.putText(img_draw, "Tracking fail.", (100,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
                
    trackerName = tracker.__class__.__name__
    
    cv2.putText(img_draw, str(trackerIdx) + ":"+trackerName , (100,20), \
                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2,cv2.LINE_AA)


    cv2.imshow(win_name, img_draw)
    
    key = cv2.waitKey(delay) & 0xff

    if key == ord(' ') or (video_src != 0 and isFirst): 
        isFirst = False
        roi = cv2.selectROI(win_name, frame, False) 
        if roi[2] and roi[3]:    
            tracker = trackers[trackerIdx]()   
            isInit = tracker.init(frame, roi)
    elif key in range(48, 56): 
        trackerIdx = key-48    
        if bbox is not None:
            tracker = trackers[trackerIdx]() 
            isInit = tracker.init(frame, bbox) 
    elif key == 27 : 
        break
    
else:
    print( "Could not open video")
cap.release()
cv2.destroyAllWindows()