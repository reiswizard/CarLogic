import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import time
import datetime

# import sys
# sys.path.append(r'/home/pi/picar-x/lib')
# from picarx import Picarx

_SHOW_IMAGE = False


# class CarControl(object):
#     car_control = Picarx()
#
#     def car_camera_angle1(self, angle):
#         self.set_camera_servo1_angle(angle)
#
#     def car_camera_angle2(self, angle):
#         self.set_camera_servo2_angle(angle)
#
#     def car_control(self, angle, speed):
#         self.set_dir_servo_angle(angle)
#         self.forward(speed)


class PiCar(object):
    def __init__(self, car=None):
        logging.info('Creating a HandCodedLaneFollower...')
        self.car = car
        self.current_steering_angle = 0
        self.speed = 0

        # line color green
        self.green_line = np.array([[60, 70, 50], [100, 255, 255]])
        self.yellow_line = np.array([[30, 60, 150], [50, 255, 255]])
        self.white_line = np.array([[0, 0, 140], [115, 40, 255]])


#####################
# drive mechanic
#####################
def drive(video_file):
    ######
    # function for tracking
    file = open("time.txt", "w")
    ######
    cap = cv2.VideoCapture(video_file)
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)

    # skip first second of video.
    for i in range(3):
        _, frame_blur = cap.read()

    # video recording
    video_type = cv2.VideoWriter_fourcc(*'XVID')
    video_overlay = cv2.VideoWriter("%s_overlay.avi" % video_file, video_type, 30, (640, 480))

    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
            # for performance recording
            t1 = time.time()

            ######################
            # start image processing, roi is region of interest
            ######################
            color_in_roi = detect_color_in_roi(frame_blur, car.yellow_line, car.green_line)
            edge_in_roi = detect_edge(color_in_roi)
            lines_segments_in_roi = detect_line(edge_in_roi)
            lanes_in_roi = average_slope_intercept(lines_segments_in_roi)
            ######################
            # end image processing

            ######################
            # determine car axis angle
            ######################
            new_steering_angle = compute_steering_angle(lanes_in_roi)
            stabilized_steering_angle = stabilize_steering_angle(car.current_steering_angle,
                                                                 new_steering_angle, len(lanes_in_roi))
            car.current_steering_angle = stabilized_steering_angle
            ######################
            # end angle
            ######################

            ######################
            # test
            ######################
            left_lane_x, right_lane_x, test_angle = compute_lanes(color_in_roi)
            lane_img = display_lanes(frame_blur, left_lane_x, right_lane_x)
            cv2.imshow("Lane to Detect", lane_img)
            print(new_steering_angle, test_angle)

            # end processing time
            t2 = time.time()

            ######################
            # car control
            ######################
            if len(lanes_in_roi) == 0:
                car.speed = 0
                # stop car
                # forward(0)
            else:
                car.speed = 20
                # forward(20)
                # angle(car.current_steering_angle)

            # visualization part begin
            cv2.imshow("Color Filtered", color_in_roi)
            cv2.imshow("Edge Detection", edge_in_roi)
            lane_lines_img = display_lines(frame_blur, lines_segments_in_roi)
            heading_line_img = display_heading_line(lane_lines_img, car.current_steering_angle,
                                                    line_color=(0, 0, 255),
                                                    line_width=5)
            video_overlay.write(heading_line_img)
            cv2.imshow("Road with Lane line", heading_line_img)
            string_value_time = str(t2 - t1)
            string_angle = str(new_steering_angle)
            string_lanes = str(lanes_in_roi)
            print(string_value_time + ", " + string_angle + " Grad")
            # file.write(string_value_time +", " + string_angle + " Grad, offset Center " + string_offset + "\n")
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # visualization part end
    finally:
        car.current_steering_angle = 0
        car.speed = 0
        cap.release()

        # visualization part release
        video_overlay.release()
        cv2.destroyAllWindows()
    # return 0


############################
# logic in use
############################
# marking all edges
def detect_edge(img):
    return cv2.Canny(img, 200, 400)


# detect color
def detect_color_in_roi(img, color_range1, color_range2, min_detected_color=1000):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # select priority color
    mask1st = cv2.inRange(hsv, color_range1[0], color_range1[1])
    mask_roi_l = roi_l(mask1st)
    mask_roi_r = roi_r(mask1st)
    detected_prio_color_l = np.sum(mask_roi_l == 255)
    detected_prio_color_r = np.sum(mask_roi_r == 255)

    # select second priority
    if detected_prio_color_l < min_detected_color or detected_prio_color_r < min_detected_color:
        mask2nd = cv2.inRange(hsv, color_range2[0], color_range2[1])
    if detected_prio_color_l < min_detected_color:
        mask_roi_l = roi_l(mask2nd)
    if detected_prio_color_r < min_detected_color:
        mask_roi_r = roi_r(mask2nd)

    # create end ROI
    mask_white_green = cv2.bitwise_or(mask_roi_l, mask_roi_r)
    end_mask = region_of_interest(mask_white_green)

    return end_mask


# take out zone of no interest left
def roi_l(img, width_sliding=0):
    height, width = img.shape
    mask = np.zeros_like(img)

    # define the lane area
    polygon = np.array([[
        (0, height * 1 / 4),  # top left
        ((1 / 2 * width - width_sliding), height * 1 / 4),  # top right
        ((1 / 2 * width - width_sliding), height * 3 / 4),  # bottom right
        (0, height * 3 / 4),  # bottom left
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


# take out zone of no interest right
def roi_r(img, width_sliding=0):
    height, width = img.shape
    mask = np.zeros_like(img)

    # define the lane area
    polygon = np.array([[
        ((1 / 2 * width + width_sliding), height * 1 / 4),  # top left
        (width, height * 1 / 4),  # top right
        (width, height * 3 / 4),  # bottom right
        ((1 / 2 * width + width_sliding), height * 3 / 4),  # bottom left
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


# take out zone of no interest final
def region_of_interest(img, height_sliding=0):
    height, width = img.shape
    mask = np.zeros_like(img)

    # define the lane area
    polygon = np.array([[
        (0, height * 1 / 4 + height_sliding),  # top left
        (width, height * 1 / 4 + height_sliding),  # top right
        (width, height * 3 / 4),  # bottom right
        (0, height * 3 / 4),  # bottom left
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


def sliding_window(img, height_start=480, height_sliding=1):
    height, width = img.shape
    mask = np.zeros_like(img)

    # define the lane area
    polygon = np.array([[
        (0, height_start + height_sliding),  # top left
        (width, height_start + height_sliding),  # top right
        (width, height_start),  # bottom right
        (0, height_start),  # bottom left
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


# detect lane segments
def detect_line(img):
    # tuning min_threshold, minLineLength, maxLineGap
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    lines = cv2.HoughLinesP(img, rho, angle, min_threshold, minLineLength=8,
                            maxLineGap=4)

    return lines


# make to lane lines out of all detected lines segments with line equation
def average_slope_intercept(line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines

    # used resolution, here hard coded, else with img
    width = 640
    height = 480

    left_fit = []
    right_fit = []

    # lanes roi
    lanes_roi_restriction = 1 / 3
    left_region_boundary = width * (1 - lanes_roi_restriction)
    right_region_boundary = width * lanes_roi_restriction

    # calculating all linear function for all detected line segments
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                # skipping vertical line segment, result of roi cut
                continue
            # line equation
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    # creating the average lane on right and left roi when they exist
    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(left_fit_average))
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(right_fit_average))

    return lane_lines


def make_points(line):
    width = 640
    height = 480
    slope, intercept = line
    y1 = height  # bottom points
    y2 = int(y1 * 1 / 2)  # middle points

    # bound the coordinates within the frame using line equalation
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))

    return [[x1, y1, x2, y2]]


# steering wheel angle
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

    # y-achse entending
    # a little tweaking for more stability, because the camera is very near the lane
    # in normal calculating 1/2 like the chosen parameter in make points is the correct points
    y_offset = int(height * 3 / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(math.degrees(angle_to_mid_radian))

    if angle_to_mid_deg < -25:
        return -25
    elif angle_to_mid_deg > 25:
        return 25
    else:
        return angle_to_mid_deg


# correcting angle reduction, to take out false detection and to sudden correction changing direction
def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lanes,
                             max_angle_deviation_two_lines=3, max_angle_deviation_one_lane=1):
    if num_of_lanes == 2:
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle +
                                        max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    return stabilized_steering_angle


#################
# test logic
#################
# histogram
def compute_lanes(color_filtered_in_roi, min_detected_color=500):
    lanes = 0
    left_lane = []
    right_lane = []
    plot_y = np.array([float(x) for x in range(360, 120, -5)])

    detected_color = np.sum(color_filtered_in_roi == 255)

    if detected_color > min_detected_color:
        i = 360
        while i > 120:
            hist_img = sliding_window(color_filtered_in_roi, i, 5)
            hist = np.sum(hist_img, axis=0)
            max_left = np.nanargmax(hist[0:320])
            max_right = np.nanargmax(hist[320:640]) + 320

            if 1 < max_left < 319:
                left_lane.append((max_left, i))
            if 320 < max_right < 639:
                right_lane.append((max_right, i))
            i -= 5
    else:
        return left_lane, right_lane, 0

    if len(left_lane) > 0:
        x_coords_left = np.array(left_lane)[:, 0]
        y_coords_left = np.array(left_lane)[:, 1]
        left_fit = np.polynomial.polynomial.polyfit(y_coords_left, x_coords_left, 2)
        left_fit_x = left_fit[2] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[0]
        left_lane = np.array((left_fit_x, plot_y)).T
        x1 = left_lane[-1][0]
    else:
        x1 = 0

    if len(right_lane) > 0:
        x_coord_right = np.array(right_lane)[:, 0]
        y_coords_right = np.array(right_lane)[:, 1]
        right_fit = np.polynomial.polynomial.polyfit(y_coords_right, x_coord_right, 2)
        right_fit_x = right_fit[2] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[0]
        right_lane = np.array((right_fit_x, plot_y)).T
        x2 = right_lane[-1][0]
    else:
        x2 = 640

    x_offset = (x1 + x2) / 2 - 320
    angle_to_mid_radian = math.atan(x_offset / 480)
    angle_to_mid_deg = int(math.degrees(angle_to_mid_radian))

    return left_lane, right_lane, angle_to_mid_deg


#################
# old logic
#################
# perspective correction
# def perspective_correction(img):
#     src_pts = np.array([[180, 105],  # top left
#                         [460, 105],  # top right
#                         [640, 220],  # bottom right
#                         [0, 220]],  # bottom left
#                        dtype=np.float32)
#     dst_pts = np.array([[0, 0], [640, 0], [640, 115], [0, 115]], dtype=np.float32)
#     perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
#     return cv2.warpPerspective(img, perspective, (640, 115), flags=cv2.INTER_LANCZOS4)
#
#
# def length_of_line_segment(line):
#     x1, y1, x2, y2 = line
#     return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

############################
# no logic
############################
def show_image(title, img, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, img)


def display_lines(frame, lines, line_color=(0, 0, 255), line_width=5):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(img, steering_angle, line_color=(255, 0, 0), line_width=5):
    heading_image = np.zeros_like(img)
    width = 640
    height = 480

    steering_angle_radian = (steering_angle + 90) / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1 - 100), (x2, y2 - 100), line_color, line_width)
    heading_image = cv2.addWeighted(img, 0.8, heading_image, 1, 1)

    return heading_image


def display_lanes(img, lane_left, lane_right):
    i = 0
    while i < len(lane_left) - 1:
        cv2.line(img, (int(lane_left[i][0]), int(lane_left[i][1])),
                 (int(lane_left[i + 1][0]), int(lane_left[i + 1][1])),
                 (0, 0, 255), 2)
        i += 1

    i = 0
    while i < len(lane_right) - 1:
        cv2.line(img, (int(lane_right[i][0]), int(lane_right[i][1])),
                 (int(lane_right[i + 1][0]), int(lane_right[i + 1][1])),
                 (0, 0, 255),
                 2)
        i += 1

    return img


############################
# Test Function
############################
def test_photo(file):
    image = cv2.imread(file)
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # drawing roi
    img_roi = cv2.rectangle(image_blur, (2, 120), (638, 360), (255, 0, 0), 3)
    img_roi = cv2.line(img_roi, (319, 120), (319, 360), (255, 0, 0), 3)
    cv2.imshow("Region of Interest", img_roi)
    cv2.imwrite('region of Interest.jpg', img_roi)

    # image processing
    color_in_roi = detect_color_in_roi(image_blur, car.yellow_line, car.green_line)
    edge_in_roi = detect_edge(color_in_roi)
    lines_segments_in_roi = detect_line(edge_in_roi)
    lanes_in_roi = average_slope_intercept(lines_segments_in_roi)

    # determine car axis angle
    steering_angle = compute_steering_angle(lanes_in_roi)

    lane_lines_img = display_lines(image_blur, lines_segments_in_roi)
    heading_line_img = display_heading_line(lane_lines_img, steering_angle, line_color=(0, 0, 255), line_width=5)

    # Histogram
    # pixel in colum
    left_lane_x, right_lane_x, test_angle = compute_lanes(color_in_roi)
    lane_img = display_lanes(image_blur, left_lane_x, right_lane_x)
    print(steering_angle, test_angle)

    show_image('Original', image_blur, True)
    show_image('Color Selection', color_in_roi, True)
    show_image('Edge', edge_in_roi, True)
    cv2.imshow("Lane Lines", lane_lines_img)
    cv2.imshow("Heading", heading_line_img)
    cv2.imwrite('Line Detected.jpg', lane_img)
    cv2.imwrite('Hough Detected.jpg', lane_lines_img)
    cv2.imwrite('Color selection.jpg', color_in_roi)
    cv2.imwrite('Edge selection.jpg', edge_in_roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(video_file):
    file = open("time.txt", "w")
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)
    cap = cv2.VideoCapture(video_file)

    # skip first second of video.
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    video_overlay = cv2.VideoWriter("%s_overlay.avi" % cap, video_type, 20.0, (640, 480))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            t1 = time.time()

            color = detect_color_in_roi(frame, car.white_line, car.green_line)
            cv2.imshow("Color", color)
            interest = region_of_interest(color)
            edge = detect_edge(interest)
            lines = detect_line(edge)
            lanes = average_slope_intercept(lines)

            new_steering_angle = compute_steering_angle(lanes)
            # new_steering_angle = angle_slope(lines)
            # new_steering_angle = angle_per_funtion(lines)

            stabilized_steering_angle = stabilize_steering_angle(car.current_steering_angle,
                                                                 new_steering_angle, len(lanes))
            car.current_steering_angle = stabilized_steering_angle
            print(new_steering_angle, stabilized_steering_angle)

            lane_lines_img = display_lines(frame, lines)
            heading_line_img = display_heading_line(lane_lines_img, car.current_steering_angle, line_color=(0, 0, 255),
                                                    line_width=5)

            # cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, steering_angle), frame)
            # cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), heading_line_img)
            video_overlay.write(heading_line_img)
            cv2.imshow("Road with Lane line", heading_line_img)
            t2 = time.time()

            string_value = str(t2 - t1)
            file.write(string_value + "\n")

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()


############################
# test main
############################
if __name__ == '__main__':
    car = PiCar('PiCar')
    # drive('filename.avi')
    test_photo('test_pic/image_163_090.png')
    # test_photo('ori.jpg')
    # test_video('filename.avi')
