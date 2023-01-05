import cv2
import numpy as np

img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
text = 'OpenCV Programming'
org = (50,100)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, text, org, font, 1, (255,0,0), 2) #img, text, 시작위치, 폰트, scale, 색깔, 두께

size, baseLine = cv2.getTextSize(text, font, 1, 2) #문자열이 출력되는 크기, 가장 하단의 텍스트 위치를 기준으로 하는 기준선(baseline)의 y 좌표, 아마 빠져나온 문자말하는 듯
print('size =', size)
print('baseLine =', baseLine)
cv2.rectangle(img, org, (org[0] + size[0], org[1] - size[1]),(0,0,255)) #img, org(왼쪽아래), 오른쪽위, 색깔
cv2.circle(img, org, 3, (0,255,0), 2) #img,중심점,반지름,색깔,두께

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()