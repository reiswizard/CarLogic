import cv2
import numpy as np

image = cv2.imread('test_pic/image_005_090.png')

b = image.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0


g = image.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = image.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0


# RGB - Blue
cv2.imshow('B-RGB', b)

# RGB - Green
cv2.imshow('G-RGB', g)

# RGB - Red
cv2.imshow('R-RGB', r)

hori = np.concatenate((b, g, r), axis=1)
cv2.imwrite('horiRGB.jpg', hori)
cv2.imshow('HORIZONTAL', hori)


cv2.waitKey(0)