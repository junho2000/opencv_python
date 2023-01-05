import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

cap = cv2.VideoCapture(0)
fig = plt.figure(figsize=(10,6))
fig.canvas.manager.set_window_title('Video Capture')
plt.axis('off')

def init():
  global im
  retval, frame = cap.read()
  im = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def updateFrame(k): #k는 애니메이션 프레임 번호 0,1,2 ...
  retval, frame = cap.read()
  if retval:
    im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

ani = animation.FuncAnimation(fig, updateFrame, init_func=init, 
                              interval=50) #프레임 50밀리초
plt.show()
if cap.isOpened():
  cap.release()