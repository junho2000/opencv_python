import cv2
import numpy as np

src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/T.jpg', cv2.IMREAD_GRAYSCALE)
ret, A = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY) #임계값을 적용하여 이진영상 생성 
skel_dst = np.zeros(src.shape, np.uint8)
B = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3)) #커널 역할을 하는 구조요소 B생성\
# 1 1 1            0 1 0
# 1 1 1            1 1 1
# 1 1 1 for RECT   0 1 0 for CROSS
done = True
while done: #골격화
    erode = cv2.erode(A,B) #A를 구조B로 침식
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, B) #오프닝 적용
    tmp = cv2.subtract(erode, opening)
    skel_dst = cv2.bitwise_or(skel_dst, tmp)
    A = erode.copy() #침식했을 때 전부 값이 0이되면 밑의 문장에 의해 반복문 종료 이때 골격만 남게 됨
    done = cv2.countNonZero(A) != 0 #A가 공집합이면 반복문 종료
    
cv2.imshow('src', src)
cv2.imshow('skel_dst', skel_dst)
cv2.waitKey()
cv2.destroyAllWindows()