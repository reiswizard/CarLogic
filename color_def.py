import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def nothing(x):
    pass


# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h0, s0, v0 = 100, 100, 100
h1, s1, v1 = 100, 100, 100

# Creating track bar
cv2.createTrackbar('h0', 'result', 0, 180, nothing)
cv2.createTrackbar('s0', 'result', 0, 255, nothing)
cv2.createTrackbar('v0', 'result', 0, 255, nothing)

cv2.createTrackbar('h1', 'result', 0, 180, nothing)
cv2.createTrackbar('s1', 'result', 0, 255, nothing)
cv2.createTrackbar('v1', 'result', 0, 255, nothing)

while 1:

    _, frame = cap.read()

    # converting to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get info from track bar and appy to result
    h0 = cv2.getTrackbarPos('h0', 'result')
    s0 = cv2.getTrackbarPos('s0', 'result')
    v0 = cv2.getTrackbarPos('v0', 'result')

    h1 = cv2.getTrackbarPos('h1', 'result')
    s1 = cv2.getTrackbarPos('s1', 'result')
    v1 = cv2.getTrackbarPos('v1', 'result')

    # Normal masking algorithm
    lower_range = np.array([h0, s0, v0])
    upper_range = np.array([h1, s1, v1])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('result', result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
