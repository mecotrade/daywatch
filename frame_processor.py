import numpy as np
import datetime
import cv2
import os
import sys
import json


class FrameProcessor:

    _WINDOW_LABEL = 'Security Feed'

    def __init__(self, detector, recognizer, logger, class_colors, class_names, background_names, background_boxes,
                 min_class_conf, background_overlap, screenshot_dir, quality, raw_screenshot_dir):

        self.detector = detector
        self.recognizer = recognizer
        self.logger = logger
        self.class_colors = class_colors
        self.class_names = class_names
        self.background_names = background_names
        self.background_boxes = background_boxes
        self.min_class_conf = min_class_conf
        self.background_overlap = background_overlap
        self.screenshot_dir = screenshot_dir
        self.quality = min(100, max(1, quality))
        self.raw_screenshot_dir = raw_screenshot_dir

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

            # inspect subframes with movement
            boxes = self.recognizer(frame, rects)
            description = []
            for box in boxes:
                top_classes = [(self.class_names[c], box[5 + c]) for c in np.argsort(box[5:])[::-1]
                               if box[5 + c] > self.min_class_conf]
                if len(top_classes) > 0:
                    name = top_classes[0][0]
                    background = name in self.background_names
                    blind_boxes = self.background_boxes[name] if name in self.background_boxes else None
                    if not background and blind_boxes is not None:
                        for blind_box in blind_boxes:
                            overlap_area = (max(box[0], blind_box[0]) -
                                            min(box[0] + box[2], blind_box[0] + blind_box[2])) * \
                                           (max(box[1], blind_box[1]) - min(box[1] + box[3], blind_box[1] + blind_box[3]))
                            # which part of detected box is contained in blind box
                            overlap_part = overlap_area / ((box[2] + 1) * (box[3] + 1))
                            if overlap_part > self.background_overlap:
                                background = True
                                self.logger.debug('[%s] background (%.3f) %s: %s' %
                                                  (FrameProcessor.get_filename(current_time),
                                                   overlap_part, name, box[:4]))
                                break
                    if not background:
                        screenshot = True

                    description += [(top_classes, box[:5])]

            if screenshot:

                description_ext = {'movements': rects,
                                   'objects': [(dict(top_classes), box) for top_classes, box in description]}

                if self.raw_screenshot_dir:
                    self.save_raw_frame_with_description(current_time, frame, description_ext)

                for top_classes, box in description:
                    name = top_classes[0][0]
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  self.class_colors[name], 1)
                    cv2.putText(frame, '%.1f%%' % (box[4] * 100,), (box[0] + box[2] // 2 - 20, box[1]),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                self.class_colors[name], 1)

                    for i, (n, c) in enumerate(top_classes):
                        cv2.putText(frame, '%s: %.1f%%' % (n, c * 100),
                                    (box[0] + box[2] + 5, box[1] + 20 * (i + 1)),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    self.class_colors[n], 1)

                self.logger.info('[%s] %s' % (FrameProcessor.get_filename(current_time), description_ext))

            # draw rectangles around subframes with movement
            for x, y, w, h in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            self.logger.debug('movement detected at %s' % rects)

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
        if (screenshot or key == ord('s')) and self.screenshot_dir:
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

    @staticmethod
    def get_filename(time):
        return time.strftime('%Y%m%d%H%M%S%f')

    def save_frame(self, frame_time, frame):

        today_screenshot_dir = os.path.join(self.screenshot_dir, frame_time.strftime('%Y%m%d'))
        if not os.path.exists(today_screenshot_dir):
            os.makedirs(today_screenshot_dir)
            self.logger.info('Created screenshot directory %s' % today_screenshot_dir)

        cv2.imwrite(os.path.join(today_screenshot_dir, '%s.jpg' % FrameProcessor.get_filename(frame_time)),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])

    def save_raw_frame_with_description(self, frame_time, frame, description):

        today_raw_screenshot_dir = os.path.join(self.raw_screenshot_dir, frame_time.strftime('%Y%m%d'))
        if not os.path.exists(today_raw_screenshot_dir):
            os.makedirs(today_raw_screenshot_dir)
            self.logger.info('Created raw screenshot directory %s' % today_raw_screenshot_dir)

        filename = FrameProcessor.get_filename(frame_time)

        cv2.imwrite(os.path.join(today_raw_screenshot_dir, '%s.jpg' % filename),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        with open(os.path.join(today_raw_screenshot_dir, '%s.json' % filename), 'w') as jsonfile:
            json.dump(description, jsonfile)
