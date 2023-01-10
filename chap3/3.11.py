#왼쪽마우스 더블클릭 뺴고 다 잘됨 검색해봤는데 mac에서는 안됨
import cv2
import numpy as np

def onMouse(event, x, y, flags, param): #마우스 이벤트 핸들러 함수, x,y는 마우스 위치, flags는 마우스 이벤트가 발생할 때 마우스 버튼과 함께 ctrl, shift, alt등의 키가 눌렸는지 확인
  if event == cv2.EVENT_LBUTTONDOWN: #마우스왼쪽버튼클릭
    if flags & cv2.EVENT_FLAG_SHIFTKEY: #SHIFT키와 함께
      cv2.rectangle(param[0], (x-5,y-5), (x+5,y+5), (255,0,0))
      print('event :', event)
      print('flags :', flags)
    else: #SHIFT키 없이
      cv2.circle(param[0], (x,y), 5, (255,0,0), 3)
      print('event :', event)
      print('flags :', flags)

  elif event == cv2.EVENT_RBUTTONDOWN: #마우스오른쪽버튼클릭
    cv2.circle(param[0], (x,y), 5, (0,0,255), 3)
    print('event :', event)
    print('flags :', flags)
  elif event == cv2.EVENT_LBUTTONDBLCLK: #마우스왼쪽버튼 더블 클릭
    param[0] = np.zeros(param[0].shape, np.uint8) + 255
    print('event :', event)
    print('flags :', flags)
  cv2.imshow('img',param[0])
  

img = np.zeros((512,512,3), np.uint8) + 255
cv2.imshow('img',img)
cv2.setMouseCallback('img', onMouse, [img]) #윈도우 이름, 마우스이벤트핸들러, [img]->param[0] 그냥 img로 넣어도 됨 
cv2.waitKey()
cv2.destroyAllWindows()