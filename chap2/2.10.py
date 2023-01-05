import cv2

cap = cv2.VideoCapture(0)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("frame size : ", frame_size)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out1 = cv2.VideoWriter('/Users/kimjunho/Desktop/python_workspace/videos/record0.mp4', fourcc, 20.0, frame_size)
out2 = cv2.VideoWriter('/Users/kimjunho/Desktop/python_workspace/videos/record1.mp4', fourcc, 20.0, frame_size, isColor=False)

while True:
    retval, frame = cap.read()
    if not retval:
        break
    
    out1.write(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    out2.write(gray)
    
    cv2.imshow('frame', frame)    
    cv2.imshow('gray', gray)
    
    key = cv2.waitKey(25)
    if key == 27: #esc키 누르면 녹화 종료
        break

cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
