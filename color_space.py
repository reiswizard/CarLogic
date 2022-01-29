import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('test_pic/image_005_090.png')

# b = image.copy()
# # set green and red channels to 0
# b[:, :, 1] = 0
# b[:, :, 2] = 0
#
#
# g = image.copy()
# # set blue and red channels to 0
# g[:, :, 0] = 0
# g[:, :, 2] = 0
#
# r = image.copy()
# # set blue and green channels to 0
# r[:, :, 0] = 0
# r[:, :, 1] = 0


# # RGB - Blue
# cv2.imshow('B-RGB', b)
#
# # RGB - Green
# cv2.imshow('G-RGB', g)
#
# # RGB - Red
# cv2.imshow('R-RGB', r)

# hori = np.concatenate((b, g, r), axis=1)
# cv2.imwrite('hsv.jpg', hsv)
# cv2.imshow('HSV', hsv)

#############
# cv2.imshow("Original", image)
#
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV", hsv)
# h, s, v = cv2.split(hsv)
# hsv_split = np.concatenate((h, s, v), axis=1)
# cv2.imshow("Split HSV", hsv_split)
# cv2.imwrite('hsvSplit.jpg', hsv_split)
#
# merge = cv2.merge([h, s, v])
# cv2.imshow('Re-Merge', merge)
#
# out = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
# cv2.imshow('Re-RGB', out)

##########
# ddepth = cv2.CV_16S
# kernel_size = 3
#
# cv2.imshow('Original', image)
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
# sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
# cv2.imshow('Sobel X', sobelxy)
#
# canny = cv2.Canny(img_gray, 0, 80)
# cv2.imshow('Canny', canny)
#
# laplace = cv2.Laplacian(img_gray, cv2.CV_64F)
# cv2.imshow('Laplace', laplace)
#
# slc = np.concatenate((sobelxy, laplace, canny), axis=1)
# cv2.imwrite('Sobel-Laplace-Canny.png', slc)
# cv2.imshow('Sobel-Laplace-Canny', slc)

cv2.waitKey(0)
