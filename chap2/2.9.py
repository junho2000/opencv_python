import cv2, pafy

# macOS를 사용하는 경우 파인더에서 Applications(응용프로그램)> Python3.9 폴더 (또는 사용중인 Python 버전)로 이동하여 "Install Certificates.command"파일을 더블 클릭
# pip install youtube_dl==2020.12.2
# pip install pafy

url = 'https://www.youtube.com/watch?v=YF-IWSbnWr4'
video = pafy.new(url)

print('title = ', video.title) # 영상 제목
print('video.rating = ', video.rating) # 별점
print('video.duration = ', video.duration) # 전체 길이

best = video.getbest() # 최적의 비디오 파일양식 정보
print('best.resolution', best.resolution)

cap = cv2.VideoCapture(best.url)

while True:
    retval, frame = cap.read()
    if not retval:
        break
    cv2.imshow('frame',frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cv2.imshow('edges',edges)
    key = cv2.waitKey(25)
    if key == 27:
        break

cv2.destroyAllWindows()