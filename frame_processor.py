import numpy as np
import datetime
import cv2
import os


class FrameProcessor:

    WINDOW_LABEL = 'Security Feed'

    def __init__(self, detector, recognizer, logger, class_colors, background_names, screenshot_dir, quality):

        self.detector = detector
        self.recognizer = recognizer
        self.logger = logger
        self.class_colors = class_colors
        self.background_names = background_names
        self.screenshot_dir = screenshot_dir
        self.quality = min(100, max(1, quality))

        self.multiscreen = False

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
                if name not in self.background_names:
                    screenshot = True
                for x, y, w, h, conf in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.class_colors[name], 1)
                    cv2.putText(frame, '%s %.1f%%' % (name, conf * 100), (x, y),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, self.class_colors[name], 1)
            if screenshot:
                self.logger.info('[%s] %s' % (current_time.strftime('%Y%m%d%H%M%S%f'),
                                              {n: r for n, r in objects.items()}))

        # text in the left top of the screen
        cv2.putText(frame, 'Moving object detected' if len(rects) > 0 else 'All clear!', (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 2)

        # timestamp in the left bottom of the screen
        cv2.putText(frame, current_time.strftime('%A %d %B %Y %H:%M:%S.%f')[:-3],
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        # show the frame
        if self.multiscreen:
            multiframe[:y_mid, x_mid:, :] = cv2.resize(frame, (x_mid, y_mid))
            cv2.imshow(FrameProcessor.WINDOW_LABEL, multiframe)
        else:
            cv2.imshow(FrameProcessor.WINDOW_LABEL, frame)

        key = cv2.waitKey(1) & 0xff

        # if the 'q' key is pressed, break from the loop
        if key == ord('q'):
            return False

        # switch between single screen and multiscreen modes (will take effect next frame)
        if key == ord('m'):
            self.multiscreen = not self.multiscreen

        # save frame if moving object detected or 's' key is pressed
        if screenshot or key == ord('s'):
            self.save_frame(current_time, frame)

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
