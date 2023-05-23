import cv2
import numpy as np
        
def is_middle(image):
    height, width = image.shape
    mid = width // 2
    for y in range(height):
         if image[mid,y] == 255:
             return True
    return False
    

def f(x):
    # any operation
    pass

cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/shuttle.mp4')
if (not cap.isOpened()):
    print('Error opening video')
    
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 142, 180, f)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, f)
cv2.createTrackbar("L-V", "Trackbars", 90, 255, f)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, f)
cv2.createTrackbar("U-S", "Trackbars", 246, 255, f)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, f)

cv2.setTrackbarPos("L-H", "Trackbars", 0)
cv2.setTrackbarPos("L-S", "Trackbars", 0)
cv2.setTrackbarPos("L-V", "Trackbars", 219)
cv2.setTrackbarPos("U-H", "Trackbars", 180)
cv2.setTrackbarPos("U-S", "Trackbars", 255)
cv2.setTrackbarPos("U-V", "Trackbars", 255)

while True:
    _, frame = cap.read()
    frame  = cv2.resize(frame, (600,400))
    height, width, _ = frame.shape
    mid = width // 2
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel)
    # HSV color space mask
    
    print(is_middle(mask))
    
    for y in range(height):
        mask[y,mid] = 255
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(20)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()