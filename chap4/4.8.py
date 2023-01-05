import cv2

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
rects = cv2.selectROIs('selectROIs', src, False, True) 
#selectROIs윈도우에, src영상을 표시, False로 선택영역에 격자를 표시x, True로 마우스 클릭 위치 중심을 기준으로 드래그 
#스페이스바/엔터키르 눌러 반복적으로 ROI영역을 지정, Esc로 선택 종료
print('rects =', rects)
#열, 행, 가로크기, 세로크기


for r in rects:
    cv2.rectangle(src, (r[0], r[1]),
                  (r[0] + r[2], r[1] + r[3]), 255)
    #이미지파일, 시작점(x,y), 종료점(x,y), 색상
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()