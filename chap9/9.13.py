import cv2
import numpy as np

src1 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band.png')
src2 = cv2.imread('/Users/kimjunho/Desktop/OpenCV_study/pictures/band3.png')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY) #Query img (찾는 이미지)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY) #train img

siftF = cv2.SIFT_create() #SIFT 객체 생성 -> Query img의 각 특징점에 대해 매칭하는 train img의 특징점을 디스크립터를 이용해 찾는 것이 목적
kp1, des1 = siftF.detectAndCompute(img1, None) #특징점과 디스크립터 계산 
kp2, des2 = siftF.detectAndCompute(img2, None)

bf = cv2.BFMatcher() #BruteForce 매칭 객체 생성
matches = bf.knnMatch(des1, des2, k=2) #des1에서 des2로의 matches 계산

print('len(matches) =', len(matches))
for i, m in enumerate(matches[:3]):
    for j, n in enumerate(m): #예를 들어 아래의 결과를 보면 매칭이 2개이다 ex) matches[0][0].distance < matches[0][1].distance
        print('matches[{}][{}] = (queryIdx:{}, trainIdx:{}, distance:{})'.format(i,j,n.queryIdx, n.trainIdx,n.distance))

dst = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=0) #matches를 그림
cv2.imshow('dst', dst)

nndrRatio = 0.45
good_matches = [f1 for f1, f2 in matches if f1.distance < nndrRatio * f2.distance] #d1/d2 가 작을수록 좋은 매칭 -> 조건에 맞으면 첫번째 매칭 f1을 good_matches에 저장

print('len(good_matches) =', len(good_matches))
if len(good_matches) < 5:
    print('sorry, too small good matches')
    exit()
    
src1_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
src2_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

H, mask = cv2.findHomography(src1_pts, src2_pts, cv2.RANSAC, 2.0) # src1를 src2에 겹치게 변환시켜줄 행렬 H를 구함
mask_matches = mask.ravel().tolist()

h, w = img1.shape
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
pts2 = cv2.perspectiveTransform(pts, H) #img1 size x M => img2
src2 = cv2.polylines(src2, [np.int32(pts2)], True, (255,0,0), 2)

draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=mask_matches, flags=2)
dst2 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)

cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()