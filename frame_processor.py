import numpy as np
import datetime
import cv2
import os
import sys


class FrameProcessor:

    _WINDOW_LABEL = 'Security Feed'

    def __init__(self, detector, recognizer, logger, class_colors, background_names, background_boxes,
                 background_overlap, screenshot_dir, quality):

        self.detector = detector
        self.recognizer = recognizer
        self.logger = logger
        self.class_colors = class_colors
        self.background_names = background_names
        self.background_boxes = background_boxes
        self.background_overlap = background_overlap
        self.screenshot_dir = screenshot_dir
        self.quality = min(100, max(1, quality))

        self.multiscreen = False
        self.show_background = False
        self.show_background_class = None

        self.logger.info('Python version: %s' % sys.version)
        self.logger.info('OpenCV version: %s' % cv2.__version__)

    def __call__(self, frame):

        current_time = datetime.datetime.now()
        screenshot = False

        if self.multiscreen:
            multiframe = np.zeros(frame.shape, dtype=np.uint8)
            x_mid, y_mid = frame.shape[1] // 2, frame.shape[0] // 2
            multiframe[:y_mid, :x_mid, :] = cv2.resize(frame, (x_mid, y_mid))
            rects, frame_delta, frame_binary = self.detector(frame)
            frame_delta_small = cv2.resize(frame_delta, (x_mid, y_mid))
            frame_binary_small = cv2.resize(frame_binary, (x_mid, y_mid))
            for channel in range(frame.shape[2]):
                multiframe[y_mid:, :x_mid, channel] = frame_delta_small
                multiframe[y_mid:, x_mid:, channel] = frame_binary_small
        else:
            rects, _, _ = self.detector(frame)

        if rects:

            # draw rectangles around subframes with movement
            for x, y, w, h in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            self.logger.debug('movement detected at %s' % rects)

            # inspect subframes with movement
            objects = self.recognizer(frame, rects)
            for name, boxes in objects.items():
                background_name = name in self.background_names
                blind_boxes = self.background_boxes[name] if name in self.background_boxes else None
                for x, y, w, h, conf in boxes:
                    background = background_name
                    if not background and blind_boxes is not None:
                        for blind_box in blind_boxes:
                            overlap_area = (max(x, blind_box[0]) - min(x + w, blind_box[0] + blind_box[2])) * \
                                           (max(y, blind_box[1]) - min(y + h, blind_box[1] + blind_box[3]))
                            # which part of detected box is contained in blind box
                            overlap_part = overlap_area / ((w + 1) * (h + 1)) > 0.5
                            if overlap_part > self.background_overlap:
                                background = True
                                self.logger.debug('[%s] background (%.3f) %s: %s' %
                                                  (current_time.strftime('%Y%m%d%H%M%S%f'),
                                                   overlap_part, name, [x, y, w, h]))
                                break
                    if not background:
                        screenshot = True

                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.class_colors[name], 1)
                    cv2.putText(frame, '%s %.1f%%' % (name, conf * 100), (x, y),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, self.class_colors[name], 1)
            if screenshot:
                self.logger.info('[%s] %s' % (current_time.strftime('%Y%m%d%H%M%S%f'),
                                              {n: r for n, r in objects.items()}))

        # text in the left top of the screen
        cv2.putText(frame, 'Moving object detected' if len(rects) > 0 else 'All clear!', (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # timestamp in the left bottom of the screen
        cv2.putText(frame, current_time.strftime('%A %d %B %Y %H:%M:%S.%f')[:-3],
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        key = cv2.waitKey(1) & 0xff

        # if the 'q' key is pressed, break from the loop
        if key == ord('q'):
            return False

        # save frame if moving object detected or 's' key is pressed
        if screenshot or key == ord('s'):
            self.save_frame(current_time, frame)

        # switch between show/hide background objects
        if key == ord('b'):
            self.show_background = not self.show_background

        if key == ord('c'):
            if self.show_background:
                names = list(self.background_boxes.keys())
                if self.show_background_class is None:
                    self.show_background_class = names[0]
                else:
                    idx = names.index(self.show_background_class)
                    if idx < 0:
                        self.logger.debug('background name %s not found in background boxes' % self.show_background_class)
                        self.show_background_class = None
                    elif idx < len(names) - 1:
                        self.show_background_class = names[idx+1]
                    else:
                        self.show_background_class = None

        # show background objects
        if self.show_background:
            if self.show_background_class in self.background_boxes:
                cv2.putText(frame, 'Background zone for %s' % self.show_background_class, (10, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                for box in self.background_boxes[self.show_background_class]:
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  self.class_colors[self.show_background_class], thickness=2)
            else:
                cv2.putText(frame, 'Background zone for all classes', (10, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                for name, boxes in self.background_boxes.items():
                    for box in boxes:
                        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                      self.class_colors[name], thickness=2)

        # add frame to multiframe
        if self.multiscreen:
            multiframe[:y_mid, x_mid:, :] = cv2.resize(frame, (x_mid, y_mid))

        cv2.imshow(FrameProcessor._WINDOW_LABEL, multiframe if self.multiscreen else frame)

        # switch between single screen and multiscreen modes (will take effect next frame)
        if key == ord('m'):
            self.multiscreen = not self.multiscreen

        return True

    def __del__(self):
        cv2.destroyAllWindows()

    def save_frame(self, frame_time, frame, prefix=''):

        if self.screenshot_dir:
            today_screenshot_dir = os.path.join(self.screenshot_dir, frame_time.strftime('%Y%m%d'))
            if not os.path.exists(today_screenshot_dir):
                os.makedirs(today_screenshot_dir)
                self.logger.info('Created screenshot directory %s' % today_screenshot_dir)

            cv2.imwrite(os.path.join(today_screenshot_dir, '%s%s.jpg' % (prefix, frame_time.strftime('%Y%m%d%H%M%S%f'))),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
