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

distT = 200
flan = cv2.FlannBasedMatcher_create()
matches = flan.radiusMatch(des1, des2, maxDistance=distT)
print('len(matches) =', len(matches))

good_matches = []
for i, radius_match in enumerate(matches):
    if len(radius_match) != 0:
        for m in radius_match:
            if m.distance<100:  #filter by distance
                good_matches.append(m)
print('len(good_matches) =', len(good_matches))

src1_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
src2_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

H, mask = cv2.findHomography(src1_pts, src2_pts, cv2.RANSAC, 3.0)
mask_matches = mask.ravel().tolist()       

h,w = img1.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
pts2 = cv2.perspectiveTransform(pts, H)
src2 = cv2.polylines(src2, [np.int32(pts2)], True, (255, 0, 0), 2)

draw_params=dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = mask_matches, flags = 2)
dst3 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)
cv2.imshow('dst3', dst3)

cv2.waitKey()
cv2.destroyAllWindows()