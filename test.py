import cv2
import numpy as np

yellow_line = np.array([[30, 60, 150], [50, 255, 255]])

image = cv2.imread('test_pic/image_300_090.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, yellow_line[0], yellow_line[1])
result = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite('test.jpg', result)