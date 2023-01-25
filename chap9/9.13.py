import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band.png')
src2 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band3.png')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

siftF = cv2.SIFT_create()
kp1, des1 = siftF.detectAndCompute(img1, None)
kp2, des2 = siftF.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

print('len(matches) =', len(matches))
for i, m in enumerate(matches[:3]):
    for j, n in enumerate(m):
        print('matches[{}][{}] = (queryIdx:{}, trainIdx:{}, distance:{})'.format(i,j,n.queryIdx, n.trainIdx,n.distance))

dst = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=0)
cv2.imshow('dst', dst)

nndrRatio = 0.3
good_matches = [f1 for f1, f2 in matches if f1.distance < nndrRatio * f2.distance]

print('len(good_matches) =', len(good_matches))
if len(good_matches) < 5:
    print('sorry, too small good matches')
    exit()
    
src1_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
src2_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

H, mask = cv2.findHomography(src1_pts, src2_pts, cv2.RANSAC, 2.0)
mask_matches = mask.ravel().tolist()

h, w = img1.shape
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
pts2 = cv2.perspectiveTransform(pts, H)
src2 = cv2.polylines(src2, [np.int32(pts2)], True, (255,0,0), 2)

draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=mask_matches, flags=2)
dst2 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)

cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()