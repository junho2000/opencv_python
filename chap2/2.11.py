import cv2
import matplotlib.pyplot as plt

def handle_key_press(event): #ESC를 눌렀을 때의 이벤트 처리 함수(이벤트 핸들러)
  if event.key == 'escape':
    cap.release()
    plt.close()
def handle_close(evt): #창을 닫았을 때의 이벤트 처리 함수
  print('Close figure!')
  cap.release()

cap = cv2.VideoCapture(0)

plt.ion() #대화 모드 설정 -> 동영상 실행 중 ESC와 창 닫기 등을 인식
fig = plt.figure(figsize=(10,6)) #fig 객체 생성과 크기 지정
plt.axis('off') #축 제거
fig.canvas.manager.set_window_title('Video Capture') #fig객체 이름 설정
fig.canvas.mpl_connect('key_press_event', handle_key_press) #연결을 끊을 때까지 이벤트를 이벤트 핸들러에 연결 
fig.canvas.mpl_connect('close_event', handle_close)

retval, frame = cap.read() #첫 프레임 캡처
im = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #컬러로 변환 후 보여주기

while True:
  retval, frame = cap.read()
  if not retval:
    break
  im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #plt.imshow보다 빠름
  fig.canvas.draw() #캔버스 갱신
  fig.canvas.flush_events() #다른 사용자 인터페이스 이벤트를 처리 (????)

if cap.isOpened():
  cap.release()
