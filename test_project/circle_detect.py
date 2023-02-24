import cv2
import numpy as np

# 1
cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/circle_detect.mp4')   #비디오 객체 cap 생성
if (not cap.isOpened()):
    print('Error opening video')


ret, src = cap.read()
h, w, c= src.shape
roi = np.zeros((h//2, w//2 , 3), dtype=src.dtype)
roi[:] = src[h//4:h//4*3, w//4:w//4*3]
print('frame.shape =', src.shape) #1080, 1104, 3
print('roi.shape =', roi.shape) #540, 552, 3

cv2.imshow('roi', roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #세로 절반 아래, 가로 1/3~2/3  
    roi[:] = frame[h//4:h//4*3, w//4:w//4*3]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blur_gray = cv2.GaussianBlur(gray, ksize=(7,7), sigmaX=0.0) #gaussian blur
    
    #검출 이미지, 검출 방법, 해상도 비율, 최소 거리, 캐니 엣지 임곗값, 중심 임곗값, 최소 반지름, 최대 반지름
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 600, param1 = 250, param2 = 10, minRadius = 30, maxRadius = 120)
    
    if circles is not None:
        circles = np.uint16(circles)
        
        for i in circles[0,:]:
            # cv2.circle(roi,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(frame,(i[0]+h//4,i[1]+w//4),i[2],(0,255,0),2)

    cv2.imshow('frame', frame)
    cv2.imshow('roi', roi)
    
    key = cv2.waitKey(20)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
    