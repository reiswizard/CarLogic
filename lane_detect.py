import cv2
import numpy as np
import math
import logging
import time
import datetime
import sys

_SHOW_IMAGE = True
test_img = cv2.imread('/test_pic/image_000_090.png')
perspective_trapezoid = None


############################
# logic
############################
# detection lane
def detect_lane(img):
    color = detect_color(img)
    interest = region_of_interest(color)
    perspective = perspective_correction(interest)
    edge = detect_edge(interest)
    lines_elements = detect_line(edge)
    lanes = average_slope_intercept(edge, lines_elements)

    return lanes


# marking all edges
def detect_edge(img):
    return cv2.Canny(img, 200, 400)


# detect color
def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # color to detect range and masking
    lower_green = np.array([60, 160, 0])
    upper_green = np.array([80, 255, 255])

    return cv2.inRange(hsv, lower_green, upper_green)


# take out zone of no interest
def region_of_interest(img):
    mask = np.zeros_like(img)

    # define the lane area
    polygon = np.array([[
        (0, 90),  # bottom left
        (640, 90),  # top right
        (640, 300),  # bottom right
        (0, 300),  # top left
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


# perspective correction
def perspective_correction(img):
    src_pts = np.array([[180, 105],  # top left
                    [460, 105],  # top right
                    [640, 220],  # bottom right
                    [0, 220]],  # bottom left
                   dtype=np.float32)
    dst_pts = np.array([[0, 0], [640, 0], [640, 115], [0, 115]], dtype=np.float32)
    perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, perspective, (640, 115), flags=cv2.INTER_LANCZOS4)


# detect lane segments
def detect_line(img):
    test_pic_line = img.copy()
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    # lines = cv2.HoughLinesP(img, rho, angle, min_threshold, np.array([]), minLineLength=8,
    #                         maxLineGap=4)
    lines = cv2.HoughLinesP(img, rho, angle, min_threshold, minLineLength=8,
                            maxLineGap=4)

    return lines


# def length_of_line_segment(line):
#     x1, y1, x2, y2 = line
#     return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def average_slope_intercept(line_segments):
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    width = 640
    height = 480

    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(right_fit_average))

    # print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def make_points(line):
    width = 640
    height = 480
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


# steering wheel
def compute_steering_angle(lane_lines):
    width = 640
    height = 480

    if len(lane_lines) == 0:
        # no lane, stop car
        return 0

    if len(lane_lines) == 1:
        # follow lines max steering angle 35
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0
        # 0.0 means car pointing to center
        # -0.03 car is centered to left
        # +0.03 means car pointing to right

        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line

    if angle_to_mid_deg < -35:
        return -35
    elif angle_to_mid_deg > 35:
        return 35
    else:
        return angle_to_mid_deg


############################
# no logic
############################
def show_image(title, img, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, img)


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=5):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(img, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(img)
    width = 640
    height = 480

    steering_angle_radian = (steering_angle+90) / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(img, 0.8, heading_image, 1, 1)

    return heading_image


############################
# Test Functions
############################
def test_photo(file):
    t1 = time.time()

    frame = cv2.imread(file)
    color = detect_color(frame)
    interest = region_of_interest(color)
    edge = detect_edge(interest)
    lines = detect_line(edge)

    number_of_white_pix = np.sum(interest == 255)

    # if number_of_white_pix > 1000:
    #     print('green')
    # else:
    #     print('blau')

    lanes = average_slope_intercept(lines)

    steering_angle = compute_steering_angle(lanes)

    lane_lines_img = display_lines(frame, lines)
    heading_line_img = display_heading_line(lane_lines_img, steering_angle, line_color=(0, 0, 255), line_width=5)

    show_image('Original', frame, True)
    show_image('Color Selection', color, True)
    show_image('Interest', interest, True)
    show_image('Edge', edge, True)
    cv2.imshow("Lane Lines", lane_lines_img)
    cv2.imshow("Heading", heading_line_img)

    t2 = time.time()
    print(t2 - t1)
    # print(lines)
    print(number_of_white_pix)
    print(lanes)
    print(steering_angle)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


############################
# test main
############################
if __name__ == '__main__':
    test_photo('test_pic/image_005_090.png')
