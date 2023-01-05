import cv2
import numpy as np

width, height = 512, 512
x, y, R = 256, 256, 60
direction = 0  # 가운데에서 오른쪽으로 시작

while True:
    key = cv2.waitKeyEx(30)
    print(key)
    if key == 0x1B:  # ESC
        break
      # up 63232, down 63233, left 63234 , right 63235
    elif key == 63235:  # right
        direction = 0
    elif key == 63233:  # down
        direction = 1
    elif key == 63234:  # left
        direction = 2
    elif key == 63232:  # up
        direction = 3

    # key = cv2.waitKey(30)
    # print(key)
    # if key == 0x1B:  # ESC
    #     break
    #   # up 63232, down 63233, left 63234 , right 63235
    # elif key == 3:  # right
    #     direction = 0
    # elif key == 1:  # down
    #     direction = 1
    # elif key == 2:  # left
    #     direction = 2
    # elif key == 0:  # up
    #     direction = 3


    if direction == 0:  # right
        x += 10
    elif direction == 1:  # down
        y += 10
    elif direction == 2:  # left
        x -= 10
    else:  # up
        y -= 10

    if x < R:  # 왼쪽 경계를 넘을려고 할때
        x = R
        direction = 0  # right
    if x > width - R:  # 오른쪽 경계를 넘을려고 할때
        x = width - R
        direction = 2  # left
    if y < R:  # 위쪽 경계를 넘을려고 할때
        y = R
        direction = 1  # down
    if y > height - R:  # 아래쪽 경계를 넘을려고 할때
        y = height - R
        direction = 3  # up

    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255
    cv2.circle(img, (x, y), R, (0, 0, 255), -1)
    cv2.imshow('img',img)

cv2.destroyAllWindows()