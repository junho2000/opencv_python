import cv2

cap = cv2.VideoCapture(0)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("frame size : ", frame_size)

while True:
  retval, frame = cap.read() #잘읽혔는지, 프레임

  if not retval:
    break
  cv2.imshow('frame',frame)

  key = cv2.waitKey(25)
  if key == 27: #esc
    break
if cap.isOpened():
  cap.release()
cv2.destroyAllWindows()