import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('test_pic3/image_005_090.png')
green_line = np.array([[60, 160, 50], [80, 255, 255]])


def test_photo(file):
    frame = cv2.imread(file)
    color = detect_color(frame, green_line)
    edge = detect_edge(frame)
    contour = detect_contour(frame)

    cv2.imshow("Original", frame)

    cv2.imshow("Color Filter", color)
    cv2.imwrite('color_filter.jpg', color)

    cv2.imshow("Edge Filter", edge)
    cv2.imwrite('edge_filter.jpg', edge)

    cv2.imshow('shapes', contour)
    cv2.imwrite('edge_filter.jpg', edge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_edge(img):
    return cv2.Canny(img, 200, 400)


def detect_contour(img):
    img_gray = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return cv2.drawContours(img, contours, -1, (0, 255, 0), 3)


def detect_color(img, color_range):
    x1, x2, x3 = color_range[0]
    y1, y2, y3 = color_range[1]
    lower_range = np.array([x1, x2 ,x3])
    upper_range = np.array([y1, y2, y3])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower_range, upper_range)


def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    # define the lane area
    polygon = np.array([[
        (0, height * 1/4),  # top left
        (width, height * 1/4),  # top right
        (width, height),  # bottom right
        (0, height),  # bottom left
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


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


if __name__ == '__main__':
    # test_photo('test_pic3/image_121_090.png')
    # car = PiCar()
    test_photo('test_pic3/image_005_090.png')