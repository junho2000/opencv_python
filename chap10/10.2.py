# #예제 10.2 배경 차영상 이동물체 검출
# import cv2
# import numpy as np

# #1
# cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/vtest.mp4')   #비디오 객체 cap 생성
# if (not cap.isOpened()):
#     print('Error opening video')

# height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) #프레임 크기 읽기

# TH = 40                             #binary threshold
# AREA_TH = 80                        #area   threshold
# bkg_gray = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/avg_gray.png', cv2.IMREAD_GRAYSCALE)
# bkg_bgr = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/avg_bgr.png')

# mode = cv2.RETR_EXTERNAL
# method = cv2.CHAIN_APPROX_SIMPLE

# #2
# t = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     t += 1
#     print('t=', t)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #2-1
#     diff_gray = cv2.absdiff(gray, bkg_gray)
# ##  ret, bImage = cv2.threshold(diff_gray, TH, 255,
# ##                              cv2.THRESH_BINARY)    

# #2-2
#     diff_bgr = cv2.absdiff(frame, bkg_bgr)
#     db, dg, dr = cv2.split(diff_bgr)
#     ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY)
#     ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY)
#     ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY)
    
#     bImage = cv2.bitwise_or(bb, bg)
#     bImage = cv2.bitwise_or(br, bImage)

#     bImage = cv2.erode(bImage, None, 5)
#     bImage = cv2.dilate(bImage, None, 5)
#     bImage = cv2.erode(bImage, None, 7)
        
# #2-3
#     image, contours, hierarchy = cv2.findContours(bImage, mode, method)
#     cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)
#     for i, cnt in enumerate(contours):
#         area = cv2.contourArea(cnt)
#         if area > AREA_TH:
#             x, y, width, height = cv2.boundingRect(cnt)
#             cv2.rectangle(frame, (x, y), (x + width, y + height),
#                             (0, 0, 255), 2)
    
#     cv2.imshow('frame', frame)
#     cv2.imshow('bImage', bImage)
#     cv2.imshow('diff_gray', diff_gray)
#     cv2.imshow('diff_bgr', diff_bgr)
#     key = cv2.waitKey(25)
#     if key == 27:
#         break
# #3
# if cap.isOpened():
#     cap.release()
# cv2.destroyAllWindows()

#예제 10.2 배경 차영상 이동물체 검출(다른 코드)
#배경과 비슷한 색깔의 객체는 구분하기 힘듬.
import cv2
import numpy as np

#1
cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/vtest.mp4')

# if not cap.isOpened():
#     print('Video open failed!')
#     sys.exit()

# 첫 프레임 배경 영상 등록
# ret, back = cap.read()
back = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/avg_bgr.png') #프레임의 평균을 back으로 지정
# if not ret:
#     print('Background image registration failed!')
#     sys.exit()
    
# 연산 속도를 높이기 위해 그레이스케일 영상으로 변환
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

# 가우시안 블러로 노이즈 제거 (모폴로지, 열기, 닫기 연산도 가능)
back = cv2.GaussianBlur(back, (0, 0), 1.0)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 현재 프레임 영상 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    
    # 차영상 구하기 $ 이진화
    # absdiff는 차 영상에 절대값
    diff = cv2.absdiff(gray, back)
    # 차이가 30이상 255(흰색), 30보다 작으면 0(검정색)
    _, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # 레이블링을 이용하여 바운딩 박스 표시
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)
    
    for i in range(1, cnt):
        x, y, w, h, s = stats[i]
        
        if s < 100:
            continue
            
        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)
    
    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()