import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band.png')
src2 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band3.png')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

#2-1
orbF = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orbF.detectAndCompute(img1, None) #특징 검출과 디스크립터 계산
kp2, des2 = orbF.detectAndCompute(img2, None)

#3-1
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True) #매칭 객체 생성
matches = bf.match(des1, des2) #매칭 계산

#4
matches = sorted(matches, key=lambda m: m.distance)
print('len(matches) =', len(matches))
for i, m in enumerate(matches[:3]):
    print('matches[{}] = (queryIdx:{}, trainIdx:{}, distance:{})'.format(i, m.queryIdx, m.trainIdx, m.distance))
    
minDist = matches[0].distance #최소값
good_matches = list(filter(lambda m: m.distance < 5 * minDist, matches))
print('len(good_matches) =', len(good_matches))
if len(good_matches) < 5:
    print('sorry, too small good matches')
    exit()
    
dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
cv2.imshow('dst', dst)

#5
src1_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
src2_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches])

H, mask = cv2.findHomography(src1_pts, src2_pts, cv2.RANSAC, 3.0)
mask_matches = mask.ravel().tolist() #list(mask.flatten())

#6
h, w = img1.shape
pts = np.float32([[0,0], [0,h-1], [w-1, h-1], [w-1,0]]).reshape(-1,1,2)
pts2 = cv2.perspectiveTransform(pts, H)
print('H =', H)
src2 = cv2.polylines(src2, [np.int32(pts2)], True, (255,0,0), 2)

draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=mask_matches, flags=2)
dst2 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)

cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()