import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('test_pic/image_005_090.png')

def callback(x):
    print(x)


canny = cv2.Canny(image, 85, 255)
cv2.namedWindow('image')
cv2.createTrackbar('L', 'image', 0, 255, callback)
cv2.createTrackbar('U', 'image', 0, 255, callback)

while 1:
    # numpy_horizontal_concat = np.concatenate((image, canny), axis=1)  # to display image side by side
    cv2.imshow('image', canny)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(image, l, u)

cv2.waitKey(0)