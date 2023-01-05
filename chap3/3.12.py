#mac에서는 안됨
import cv2
import numpy as np

def onChange(pos): #트랙바 핸들러
  global img
  r = cv2.getTrackbarPos('R', 'img') #슬라이더의 위치를 리턴
  g = cv2.getTrackbarPos('G', 'img')
  b = cv2.getTrackbarPos('B', 'img')
  img[:] = (b,g,r) 
  cv2.imshow('img',img)

img = np.zeros((512,512,3), np.uint8)
cv2.imshow('img',img)

#트랙바 생성
cv2.createTrackbar('R', 'img', 0, 255, onChange) #슬라이더 초기 위치 0,최대값 255,이벤트핸들러함수
cv2.createTrackbar('G', 'img', 0, 255, onChange)
cv2.createTrackbar('B', 'img', 0, 255, onChange)

#트랙바 위치 초기화
cv2.setTrackbarPos('B', 'img', 255) 

cv2.waitKey()
cv2.destroyAllWindows()