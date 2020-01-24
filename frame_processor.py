import numpy as np
import datetime
import cv2
import os
import sys
import json


class FrameProcessor:

    _WINDOW_LABEL = 'Security Feed'

    _SYSTEM_COLOR = (0, 0, 255)
    _GRID_COLOR = (0, 255, 0)
    _BACKGROUND_COLOR = (255, 255, 255)
    _MOTION_MERGED_BOX_COLOR = (0, 0, 255)
    _MOTION_BOX_COLOR = (0, 255, 0)

    def __init__(self, detector, recognizer, onvif_connector, logger, class_colors, class_names,
                 background_names, background_boxes, min_class_conf, background_overlap,
                 screenshot_dir, quality, raw_screenshot_dir, max_screen_size):

        self.detector = detector
        self.recognizer = recognizer
        self.onvif_connector = onvif_connector
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
        self.max_screen_size = max_screen_size

        self.multiscreen = False
        self.show_background = False
        self.show_background_class = None
        self.moving = False
        self.screen_shape = None

        cv2.namedWindow(FrameProcessor._WINDOW_LABEL, cv2.WINDOW_NORMAL)

        self.logger.info('Python version: %s' % sys.version)
        self.logger.info('OpenCV version: %s' % cv2.__version__)

    def mouse_callback(self, event, x, y, flags, param):

        if self.onvif_connector is not None:
            w, h = param[1], param[0]

            if event == cv2.EVENT_LBUTTONDOWN:
                pan = -1 if x < w // 3 else 1 if x > 2*w // 3 else 0
                tilt = -1 if y < h // 3 else 1 if y > 2*h // 3 else 0
                self.onvif_connector.continuous_move(pan, tilt)
                self.moving = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.onvif_connector.stop()
                self.moving = False
                if self.detector is not None:
                    self.detector.reset()

    @staticmethod
    def add_rects_to_multiscreen(multiframe, rects, color):
        x_mid, y_mid = multiframe.shape[1] // 2, multiframe.shape[0] // 2
        for x, y, w, h in rects:
            cv2.rectangle(multiframe, (x // 2, y_mid + y // 2),
                          ((x + w) // 2, y_mid + (y + h) // 2), color, 1)
            cv2.rectangle(multiframe, (x_mid + x // 2, y_mid + y // 2),
                          (x_mid + (x + w) // 2, y_mid + (y + h) // 2), color, 1)

    def __call__(self, frame):

        if self.screen_shape is None \
                or self.screen_shape[0] != frame.shape[0] or self.screen_shape[1] != frame.shape[1]:
            self.screen_shape = frame.shape[:2]
            cv2.setMouseCallback(FrameProcessor._WINDOW_LABEL, self.mouse_callback, param=self.screen_shape)

        current_time = datetime.datetime.now()
        screenshot = False

        if self.multiscreen:
            multiframe = np.zeros(frame.shape, dtype=np.uint8)
            x_mid, y_mid = frame.shape[1] // 2, frame.shape[0] // 2
            multiframe[:y_mid, :x_mid, :] = cv2.resize(frame, (x_mid, y_mid))

        if not self.moving:

            if self.detector is not None:
                rects, (frame_delta, frame_binary, motion_rects) = self.detector(frame)
                if self.multiscreen:
                    frame_delta_small = cv2.resize(frame_delta, (x_mid, y_mid))
                    frame_binary_small = cv2.resize(frame_binary, (x_mid, y_mid))
                    for channel in range(frame.shape[2]):
                        multiframe[y_mid:, :x_mid, channel] = frame_delta_small
                        multiframe[y_mid:, x_mid:, channel] = frame_binary_small
                    FrameProcessor.add_rects_to_multiscreen(multiframe, motion_rects, FrameProcessor._MOTION_BOX_COLOR)
                    FrameProcessor.add_rects_to_multiscreen(multiframe, rects, FrameProcessor._MOTION_MERGED_BOX_COLOR)
            else:
                # if no movement detector provided, the whole frame is a single input for recognizer
                motion_rects = rects = [[0, 0, frame.shape[1], frame.shape[0]]]

            # inspect subframes with movement
            if self.recognizer is not None:
                if rects:
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
                                                   (max(box[1], blind_box[1]) -
                                                    min(box[1] + box[3], blind_box[1] + blind_box[3]))
                                    # which part of detected box is contained in blind box
                                    overlap_part = overlap_area / ((box[2] + 1) * (box[3] + 1))
                                    if overlap_part > self.background_overlap:
                                        background = True
                                        self.logger.debug('[%s] background (%.3f) %s: %s' %
                                                          (FrameProcessor.get_filename(current_time),
                                                           overlap_part, name, box[:4]))
                                        break

                            # check if recognized objects have intersection with detected motions
                            if self.detector is not None:
                                if np.max([self.detector.intersect(box, r) for r in motion_rects]) == 0:
                                    background = True

                            if self.multiscreen:
                                FrameProcessor.add_rects_to_multiscreen(multiframe, [box[:4]], self.class_colors[name])

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
                            FrameProcessor.put_text(frame, '%.1f%%' % (box[4] * 100), box[0] + box[2] // 2, box[1],
                                                    self.class_colors[name], FrameProcessor._BACKGROUND_COLOR,
                                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 'bottom center')

                            offset = 0
                            for n, c in top_classes:
                                _, h, b = FrameProcessor.put_text(frame, '%s: %.1f%%' % (n, c * 100),
                                                                  box[0] + box[2], box[1] + offset,
                                                                  self.class_colors[n],
                                                                  FrameProcessor._BACKGROUND_COLOR,
                                                                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 'top left')
                                offset += h + b
                            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                          self.class_colors[name], 1)

                        self.logger.info('[%s] %s' % (FrameProcessor.get_filename(current_time), description_ext))

            if self.detector is not None:
                # draw rectangles around subframes with movement
                for x, y, w, h in motion_rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), FrameProcessor._MOTION_MERGED_BOX_COLOR, 1)
                self.logger.debug('movement detected at %s' % rects)

                # text in the left top of the screen
                cv2.putText(frame, 'Moving object detected' if len(rects) > 0 else 'All clear!', (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, FrameProcessor._SYSTEM_COLOR, 2)

        else:
            cv2.putText(frame, 'Camera moving', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, FrameProcessor._SYSTEM_COLOR, 2)

        # timestamp in the left bottom of the screen
        FrameProcessor.put_text(frame, current_time.strftime('%A %d %B %Y %H:%M:%S.%f')[:-3], 10, frame.shape[0] - 10,
                                FrameProcessor._SYSTEM_COLOR, FrameProcessor._BACKGROUND_COLOR,
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)

        key = cv2.waitKeyEx(1)

        # if the 'q' key is pressed, break from the loop
        if key == ord('q'):
            return False

        # save frame if moving object detected or 's' key is pressed
        if (screenshot or key == ord(' ')) and self.screenshot_dir:
            self.save_frame(current_time, frame)

        # switch between show/hide background objects
        if key == ord('b'):
            self.show_background = not self.show_background

        if key == ord('c'):
            if self.show_background:
                names = list(self.background_boxes.keys())
                if names:
                    if self.show_background_class is None:
                        self.show_background_class = names[0]
                    else:
                        idx = names.index(self.show_background_class)
                        if idx < 0:
                            self.logger.debug('background name %s not found in background boxes' %
                                              self.show_background_class)
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
        if self.multiscreen and self.detector is not None:
            multiframe[:y_mid, x_mid:, :] = cv2.resize(frame, (x_mid, y_mid))
            screen = multiframe
        else:
            screen = frame

        if self.moving:

            # actual width and height of the window
            _, _, w_wnd, h_wnd = cv2.getWindowImageRect(FrameProcessor._WINDOW_LABEL)
            h, w = self.screen_shape

            width_h, width_w = h // h_wnd, w // w_wnd

            x_l, x_r = w // 3, w * 2 // 3
            y_t, y_b = h // 3, h * 2 // 3
            cv2.line(screen, (x_l, 0), (x_l, h), FrameProcessor._GRID_COLOR, width_w)
            cv2.line(screen, (x_r, 0), (x_r, h), FrameProcessor._GRID_COLOR, width_w)
            cv2.line(screen, (0, y_t), (w, y_t), FrameProcessor._GRID_COLOR, width_h)
            cv2.line(screen, (0, y_b), (w, y_b), FrameProcessor._GRID_COLOR, width_h)

        if self.max_screen_size is not None and screen.shape[1] > self.max_screen_size:
            screen = cv2.resize(screen, (self.max_screen_size * screen.shape[1] // screen.shape[0],
                                         self.max_screen_size))

        cv2.imshow(FrameProcessor._WINDOW_LABEL, screen)

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
        filename_jpeg = '%s.jpg' % filename
        filename_json = '%s.json' % filename

        cv2.imwrite(os.path.join(today_raw_screenshot_dir, filename_jpeg),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        with open(os.path.join(today_raw_screenshot_dir, filename_json), 'w') as jsonfile:
            json.dump(dict(description, file=filename_jpeg), jsonfile)

    @staticmethod
    def put_text(frame, text, x, y, color, bgcolor, font, font_scale, thickness, point_type='bottom left'):

        (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)

        if point_type == 'bottom left':
            box_bottom_left = (x, y)
            box_top_right = (x + text_width, y - text_height - baseline)
            text_bottom_left = (x, y - baseline + 1)
        elif point_type == 'bottom center':
            box_bottom_left = (x - text_width // 2, y)
            box_top_right = (box_bottom_left[0] + text_width, y - text_height - baseline)
            text_bottom_left = (box_bottom_left[0], y - baseline + 1)
        elif point_type == 'top left':
            box_bottom_left = (x, y + text_height + baseline)
            box_top_right = (x + text_width, y)
            text_bottom_left = (x, y + text_height + 1)

        cv2.rectangle(frame, box_bottom_left, box_top_right, bgcolor, cv2.FILLED)
        cv2.putText(frame, text, text_bottom_left,
                    font, fontScale=font_scale, color=color, thickness=thickness)

        return text_width, text_height, baseline
