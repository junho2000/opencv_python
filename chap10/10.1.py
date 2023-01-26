#예제 10.1 평균에 의한 배경 영상
#평균에 의한 배경영상
#동영상 재생시 매 frame을 더한 뒤 frame의 수로 나눈다.
#이동하는 물체는 지속적으로 움직여 frame에서 사라지거나 값이 계속 바뀌므로 결국 움직이지 않는 background만 남게 된다.
import cv2
import numpy as np
#1
cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/vtest.mp4')   #비디오 객체 cap 생성
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) #프레임 크기 읽기

acc_gray = np.zeros(shape= (height, width), dtype= np.float32)  #그레이스케일, 컬러 프레임 누적을 위한 acc_gray, acc_bgr 생성
acc_bgr = np.zeros(shape= (height, width, 3), dtype= np.float32)
t = 0

#2
while True:
    ret, frame = cap.read()
    if not ret:
        break
    t += 1
    print('t=', t)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#2-1
    cv2.accumulate(gray, acc_gray)  #gray를 acc_gray에 누적
    avg_gray = acc_gray / t #avg_gray: 매 frame을 gray로 변환후 더한 뒤 frame의 수로 나눈다
    dst_gray = cv2.convertScaleAbs(avg_gray) #절대값 계산후 8bit로서 표현
#2-2
    cv2.accumulate(frame, acc_bgr)  #color 영상 누적
    avg_bgr = acc_bgr / t #avg_bgr: 매 frame을 더한 뒤 frame의 수로 나눈다.
    dst_bgr = cv2.convertScaleAbs(avg_bgr) #절대값 계산후 8bit로서 표현

    cv2.imshow('frame', frame)
    cv2.imshow('dst_gray', dst_gray)
    cv2.imshow('dst_bgr', dst_bgr)
    key = cv2.waitKey(20)
    if key == 27:
        break
#3
if cap.isOpened(): cap.release()
# cv2.imwrite('/Users/kimjunho/Desktop/OpenCV_study/pictures/avg_gray.png', dst_gray)
# cv2.imwrite('/Users/kimjunho/Desktop/OpenCV_study/pictures/avg_bgr.png', dst_bgr)
cv2.destroyAllWindows()