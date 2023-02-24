import cv2
import numpy as np
import cv2, pafy

# 1
cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/test_video.mp4')   #비디오 객체 cap 생성
if (not cap.isOpened()):
    print('Error opening video')

# url = 'https://www.youtube.com/watch?v=YsPdvvixYfo&t=0s'
# video = pafy.new(url)

# print('title = ', video.title) # 영상 제목
# print('video.rating = ', video.rating) # 별점
# print('video.duration = ', video.duration) # 전체 길이

# best = video.getbest() # 최적의 비디오 파일양식 정보
# print('best.resolution', best.resolution)

# cap = cv2.VideoCapture(best.url)


ret, src = cap.read()
h, w, c= src.shape
roi = np.zeros(src.shape, dtype=src.dtype)
roi[h//2:h, w//4:w//4*3] = src[h//2:h, w//4:w//4*3]
print('frame.shape =', src.shape) #720 1280
print('roi.shape =', roi.shape) #480 640

cv2.imshow('roi', roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #세로 절반 아래, 가로 1/3~2/3  
    roi = np.zeros((h//2,w//2,3), dtype=src.dtype)
    roi[:,:] = frame[h//2:h, w//4:w//4*3]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)
    # print('edges', edges.shape) 360 x 640
    
    #필요없는 부분 마스크
    edges[:h//8,:w//10] = 0
    edges[:h//8,w//8*3:] = 0
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=30)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1+w//4,y1+h//2), (x2+w//4,y2+h//2), (0,0,255), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('edges', edges)
    
    key = cv2.waitKey(20)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.waitKey()
cv2.destroyAllWindows()