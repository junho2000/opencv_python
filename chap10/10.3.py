#예제 10.3 이동평균 moving average 배경 차영상
import cv2
import numpy as np

#1
cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/vtest.mp4')   #비디오 객체 cap 생성
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) #프레임 크기 읽기

TH = 40                             #binary threshold
AREA_TH = 80                        #area   threshold
acc_bgr = np.zeros(shape= (height, width, 3), dtype= np.float32)

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

#2
t = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    t += 1
    print('t=', t)
    blur = cv2.GaussianBlur(frame, (5, 5), 0.0)
#2-1
    if t < 50:
        cv2.accumulate(blur, acc_bgr)
        continue
    elif t == 50: #t 값이 증가할 수록 초기 백그라운드의 오검출이 줄어듬
        bkg_bgr = acc_bgr / t
#2-2: t >= 50
##    diff_bgr = cv2.absdiff(np.float32(blur).
#                            bkg_bgr).astype(np.uint8)
    diff_bgr = np.uint8(cv2.absdiff(np.float32(blur), bkg_bgr))
    db, dg, dr = cv2.split(diff_bgr)
    ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY)
    ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY)
    ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY)
    bImage = cv2.bitwise_or(bb, bg)
    bImage = cv2.bitwise_or(br, bImage) #bImage 이진영상 생성
    bImage = cv2.erode(bImage, None, 5)
    bImage = cv2.dilate(bImage, None, 5)
    bImage = cv2.erode(bImage, None, 7) #노이즈 제거
    
    cv2.imshow('bImage', bImage)
    msk = bImage.copy()
    contours, hierarchy = cv2.findContours(bImage, mode, method)
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 1) #cv2.drawContontours()를 통하여 윤곽선 검출
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt) #cv2.contourArea()를 통하여 Moving Object의 크기 추출후 특정 크기 이상의 물체만 검출
        if area > AREA_TH: #최소 면적보다 작으면 검출 x
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + width, y + height),
                            (0, 0, 255), 2)
            cv2.rectangle(msk, (x, y), (x + width, y + height),
                            255, -1) #이동물체 주위도 배경영상을 갱신하지 않도록 함
    
#2-3
    msk = cv2.bitwise_not(msk)  #이동물체 영역은 0, 배경은 255로 반전
    cv2.accumulateWeighted(blur, bkg_bgr, alpha= 0.1, mask = msk) #이동 평균 영상 0.1xblur + bkg_bgr
#Mask는 배경이 255인 곳만을 가르키고 있으므로 위에서 dst는 배경에만 영향을 받아서 계속되어 Update가 된다는 것을 알 수 있다.
    cv2.imshow('frame', frame)
    cv2.imshow('bkg_bgr', np.uint8(bkg_bgr))
    cv2.imshow('diff_bgr', diff_bgr)
    key = cv2.waitKey(25)
    if key == 27:
        break
#3
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
#장점: 현재 영상에서 추가로 다른 Object(EX) 가로등, 신호등)가 설치가 되어도 지속적인 움직임이 없으면 Background로 포함시킬 수 있다는 것