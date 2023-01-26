#예제 10.4 BackgroundSubtractor 배경과 전경 분할
import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/vtest.mp4')   #비디오 객체 cap 생성
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) #프레임 크기 읽기
#1
bgMog1 = cv2.createBackgroundSubtractorMOG2() #마할라노비스 거리 제곱 사용을 임계치로 사용
bgMog2 = cv2.createBackgroundSubtractorMOG2(varThreshold= 25, detectShadows= False) #작은 값을 사용할 수록 많은 화소를 전경으로 검출, True이면 그림자를 검출
bgknn1 = cv2.createBackgroundSubtractorKNN()
bgknn2 = cv2.createBackgroundSubtractorKNN(dist2Threshold= 1000, detectShadows= False) #화소간 거리의 임계치로 작은 값을 사용하면 많은 화소를 전경으로 검출, True이면 그림자를 검출

#2 이진영상 bImage에 모폴로지 연산, 윤곽선 검출
AREA_TH  = 80   #area threshold
def findObjectAndDraw(bImage, src):
    res = src.copy()
    bImage = cv2.erode(bImage, None, 5)
    bImage = cv2.dilate(bImage, None, 5)
    bImage = cv2.erode(bImage, None, 7)
    contours, _ = cv2.findContours(bImage,
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1, (255, 0, 0), 1)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > AREA_TH:
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(res, (x, y), (x + width, y + height),
                            (0, 0, 255), 2)
    return res

t = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    t+=1
    print('t=', t)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (5, 5), 0.0)

#3
    bImage1 = bgMog1.apply(blur)
    bImage2 = bgMog2.apply(blur)
    bImage3 = bgknn1.apply(blur)
    bImage4 = bgknn2.apply(blur)
    dst1 = findObjectAndDraw(bImage1, frame)
    dst2 = findObjectAndDraw(bImage2, frame)
    dst3 = findObjectAndDraw(bImage3, frame)
    dst4 = findObjectAndDraw(bImage4, frame)
    
##      if t == 50:
    cv2.imshow('bImage1', bImage1)
    cv2.imshow('bgMog1', dst1)
    cv2.imshow('bImage2', bImage2)
    cv2.imshow('bgMog2', dst2)
    cv2.imshow('bImage3', bImage3)
    cv2.imshow('bgKnn1', dst3)
    cv2.imshow('bImage4', bImage4)
    cv2.imshow('bgKnn2', dst4)
    key = cv2.waitKey(25)
    if key == 27:
        break
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()