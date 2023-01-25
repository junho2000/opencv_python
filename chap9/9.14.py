import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band.png')
src2 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band3.png')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

siftF = cv2.SIFT_create()
kp1, des1 = siftF.detectAndCompute(img1, None)
kp2, des2 = siftF.detectAndCompute(img2, None)
print('len(kp1) =', len(kp1))
print('len(kp2) =', len(kp2))

flan = cv2.FlannBasedMatcher_create()
matches = flan.radiusMatch(des1, des2, maxDistance=50)

def draw_key2image(kp, img):
    x, y = kp.pt
    size = kp.size
    rect = ((x, y), (size, size), kp.angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(img, [box], True, (0,255,0), 2)
    cv2.circle(img, (round(x), round(y)), round(size / 2), (255,0,0), 2)

for i, radius_match in enumerate(matches):
    if len(radius_match) != 0:
        print('i =', i)
        print('len(matches[{}] = {}'.format(i, len(matches[i])))
        
        src1c = src1.copy()
        draw_key2image(kp1[radius_match[0].queryIdx], src1c)
        src2c = src2.copy()
        for m in radius_match:
            draw_key2image(kp2[m.trainIdx], src2c)
        dst = cv2.drawMatches(src1c, kp1, src2c, kp2, radius_match, None, flags=2)

        cv2.imshow('dst', dst)
        cv2.waitKey()
cv2.waitKey()
cv2.destroyAllWindows()