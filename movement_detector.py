import numpy as np
import cv2


class MovementDetector:

    def __init__(self, min_contour_area, rectangle_separation, gray_threshold, gray_smoothing):

        self.min_contour_area = min_contour_area
        self.rectangle_separation = rectangle_separation
        self.gray_threshold = gray_threshold
        self.gray_smoothing = gray_smoothing

        self.last_gray = None

    def reset(self):

        self.last_gray = None

    def produce_rects(self, contours):

        rects = []
        for contour in contours:
            # process contour only if is is large enough
            if cv2.contourArea(contour) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                rects += [[x, y, w, h]]

        return rects

    def intersect(self, a, b):
        return not ((a[0] + a[2] <= b[0] - self.rectangle_separation) or
                    (b[0] + b[2] <= a[0] - self.rectangle_separation) or
                    (a[1] + a[3] <= b[1] - self.rectangle_separation) or
                    (b[1] + b[3] <= a[1] - self.rectangle_separation))

    def add_rect(self, rects, r):

        merge = False
        for i in range(len(rects)):
            if self.intersect(rects[i], r):
                x, y, w, h = rects[i]
                rects[i][0] = min(r[0], x)
                rects[i][1] = min(r[1], y)
                rects[i][2] = max(r[0] + r[2], x + w) - rects[i][0]
                rects[i][3] = max(r[1] + r[3], y + h) - rects[i][1]
                merge = True
                break
        if not merge:
            rects += [r]

        return rects, merge

    def merge_rects(self, rects):

        new_merge = True
        while new_merge:
            new_rects = []
            new_merge = False
            for rect in rects:
                new_rects, merge = self.add_rect(new_rects, rect)
                if merge:
                    new_merge = True
            rects = new_rects
        return rects

    def __call__(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # init last gray frame at the very first call
        if self.last_gray is None:
            self.last_gray = gray

        # compute the absolute difference between the current frame and previous frame
        frame_delta = cv2.absdiff(self.last_gray, gray)
        frame_binary = cv2.threshold(frame_delta, self.gray_threshold, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        frame_binary = cv2.dilate(frame_binary, None, iterations=2)
        
        # method returns 3 values for version 3.x of OpenCV, and 2 values for version 4.x of OpenCV
        # in both cases the contours description is in before-last position
        cnts = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[-2]

        motion_rects = self.produce_rects(cnts)
        rects = self.merge_rects(motion_rects)

        self.last_gray = (self.last_gray * self.gray_smoothing + gray * (1 - self.gray_smoothing)).astype(np.uint8)

        return rects, (frame_delta, frame_binary, motion_rects)
