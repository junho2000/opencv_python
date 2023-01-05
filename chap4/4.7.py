import cv2

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png', cv2.IMREAD_GRAYSCALE)
roi = cv2.selectROI(src) #스페이스바/엔터키를 누르면 선택 영역을 roi에 반환
print('roi', roi) #열, 행, 가로크기, 세로크기

if roi != (0,0,0,0):
    img = src[roi[1]:roi[1] + roi[3],
              roi[0]:roi[0] + roi[2]]
    #선택 영역의 roi를 img에 저장
    cv2.imshow('img', img)
    cv2.waitKey()
else:
    print('you didnt choose!!')
cv2.destroyAllWindows()