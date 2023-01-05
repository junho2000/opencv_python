import cv2
src = cv2.imread('/Users/kimjunho/Desktop/python_workspace/pictures/lenna.png')

b, g, r = cv2.split(src)
dst = cv2.merge([b,g,r])

print(type(dst))
print(dst.shape)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()