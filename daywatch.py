import cv2
import urllib.request
import numpy as np
import base64
import time
import logging
from logging.handlers import TimedRotatingFileHandler
import queue
import threading
import argparse
import os
import json
from urllib.parse import urlparse

from movement_detector import MovementDetector
from recognition_engine import RecognitionEngine
from onvif_connector import ONVIFConnector
from frame_processor import FrameProcessor


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, url, retry_interval=10):

        self.url = url
        self.retry_interval = retry_interval

        self.frames_count = 0
        self.frames_discarded = 0
        self.q = queue.Queue()
        t = threading.Thread(target=self.run)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def run(self):
        while True:
            cap = cv2.VideoCapture(self.url)
            loop = True
            try:
                while loop:
                    loop, frame = cap.read()
                    self.frames_count += 1
                    if loop:
                        while not self.q.empty():
                            try:
                                self.q.get_nowait()   # discard previous (unprocessed) frame
                                self.frames_discarded += 1
                            except queue.Queue.Empty:
                                break
                        self.q.put(frame)
            except Exception as ex:
                logger.warning('Connection failed: %s' % ex)

            cap.release()
            if self.retry_interval > 0:
                time.sleep(self.retry_interval)
            else:
                break

    def read(self):
        return self.q.get()


def get_authorization(credentials):
    base64string = base64.encodebytes(bytes(('%s:%s' % (credentials[0], credentials[1])), 'utf-8')).decode('utf-8').replace('\n','')
    return 'Basic %s' % base64string


def watch_mjpg(url, credentials, retry_interval):

    # start main loop
    loop = True
    while loop:

        try:
            request = urllib.request.Request(url)
            if credentials:
                request.add_header('Authorization', get_authorization(credentials=credentials))

            with urllib.request.urlopen(request, timeout=10) as stream:

                logger.info('Communication established')

                data = bytearray()

                while loop:

                    new_bytes = stream.read(1024)
                    if len(new_bytes) == 0:
                        time.sleep(10)
                        break

                    data.extend(new_bytes)
                    begin_marker = data.find(b'\xff\xd8')
                    end_marker = data.find(b'\xff\xd9')

                    # new frame detected
                    if begin_marker != -1 and end_marker != -1:

                        jpg = data[begin_marker:end_marker+2]
                        data = data[end_marker+2:]

                        frame = cv2.imdecode(np.array(jpg), cv2.IMREAD_COLOR)
                        loop = processor(frame)

        except Exception as ex:
            logger.warning('Connection failed: %s' % ex)
            if retry_interval > 0:
                time.sleep(retry_interval)
            else:
                loop = False


def watch(url, retry_interval):

    cap = VideoCapture(url, retry_interval)

    loop = True
    while loop:
        frame = cap.read()
        loop = processor(frame)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url',
                        help='URL of the videostream, it might contain placeholders for credentials, '
                             '{user} and {passwd}, if so, they will be filled with provided credentials')
    parser.add_argument('-c', '--credentials', nargs=2,
                        help='credential, username and password')
    parser.add_argument('-mca', '--min-contour-area', type=int, default=1000,
                        help='minimal area of the contour with detected motion')
    parser.add_argument('-gt', '--gray-threshold', type=int, default=32,
                        help='motion detection threshold, if the difference of frames at grayscale is above this '
                             'threshold, motion is detected')
    parser.add_argument('-rs', '--rectangle-separation', type=int, default=20,
                        help='mimimal distance bewteen rectangle edges for rectangles to be separated')
    parser.add_argument('-mos', '--max-output-size', type=int, default=10,
                        help='maximal possible number of detected object for each class')
    parser.add_argument('-mbc', '--min-box-conf', type=float, default=0.5,
                        help='when box confidence is above this value object is considered to be detected')
    parser.add_argument('-mcc', '--min-class-conf', type=float, default=0.5,
                        help='when class confidence is above this value object is considered to be classified')
    parser.add_argument('-iout', '--iou-threshold', type=float, default=0.5,
                        help='among all intersection boxes containing classified object those whose '
                             'Intersection Over Union (iou) part is greater than this threshold are choosen')
    parser.add_argument('-s', '--selector', default='class', choices=('box', 'class', 'f1'),
                        help='Select one box out of the cluster based on the box confidence (for "box" value), '
                             'on class confidence (for "class" value), or on both combined into f1_score '
                             'for ("f1" value)')
    parser.add_argument('-wf', '--weights_file', default='yolov3.weights',
                        help='file with YOLOv3 weights. \
                        Must be downloaded from https://pjreddie.com/media/files/yolov3.weights')
    parser.add_argument('-nf', '--names-file', default='coco.names',
                        help='text file with COCO classes names, one name per line')
    parser.add_argument('-lf', '--log-file', default='watch.log',
                        help='log file')
    parser.add_argument('-sd', '--screenshot-dir',
                        help='directory where screenshots is stored, if not set, screenshots are not saved')
    parser.add_argument('-sq', '--screenshot-quality', type=int, default=80,
                        help='JPEG quality of saved screenshot, must be an integer between 1 and 100')
    parser.add_argument('-rsd', '--raw-screenshot-dir',
                        help='when it is required to collect own dataset based on existed model predictions, '
                             'which is useful when boxes quaity is acceptable, whereas class detection quality '
                             'is not, the directory where raw images (with highest quality) together with '
                             'descriptions, one json file per image, can be set here')
    parser.add_argument('-b', '--background', default=[], nargs='*',
                        help='names of background classes, objects of such classes do not trigger screenshot')
    parser.add_argument('-bf', '--background-file',
                        help='text file or json file with background classes, objects of such classes do not trigger '
                             'screenshot, if text file, one line per class, the order is not important, if json file, '
                             'object like {"preson": [[x1, y1, w1, h1], [x2, y2, w2, h2]], "car": null}, if value for '
                             'a class name is null, all object of such class are considered as background')
    parser.add_argument('-bo', '--background-overlap', default=0.5,
                        help='if area of overlap box of detected object and background object is greater that this '
                             'value, the object is considered as background. Applied only if --background-file is '
                             'a json file with background object boxes')
    parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode')
    parser.add_argument('-m', '--mjpg', action='store_true', help='connect to MJPG stream source')
    parser.add_argument('-mss', '--max-screen-size', type=int,
                        help='maximal size of screen, if not set, original frame size is used')
    parser.add_argument('-op', '--onvif-port', default=8899,
                        help='ONVIF connection port, if not set, default value 8899 is used')
    parser.add_argument('-oc', '--onvif-credentials', nargs=2,
                        help='ONVIF connection credentials, if not set, no communication is established')
    parser.add_argument('-nd', '--no-detection', action='store_true', help='no movement detection')
    parser.add_argument('-nr', '--no-recognition', action='store_true', help='no object recognition')
    parser.add_argument('-ri', '--retry-interval', default=10,
                        help='interval between connection lost and next attempt to establish connection, '
                             'if zero, not attemp to reconnect will be done')
    parser.add_argument('-gs', '--gray-smoothing', default=0.7,
                        help='to detect motion the movement detector compares new frame with EMA of previous frames,'
                             'this is the EMA smoothing parameter')
    
    args = parser.parse_args()

    # init logger
    log_dir = os.path.dirname(args.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = TimedRotatingFileHandler(args.log_file, when='midnight')
    handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    root_logger = logging.getLogger()
    root_logger.handlers = []

    logger = logging.getLogger('watch')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(handler)

    logger.info('Start watching with following configuration:')
    for a, v in vars(args).items():
        # hide password in logs
        if ('credentials' == a or 'onvif_credentials' == a) and v:
            v = [v[0], '******']
        logger.info('  - %s: %s' % (a, v))

    # load COCO class names
    with open(args.names_file, 'r') as f:
        class_names = f.read().splitlines()
    logger.info('COCO class names: %s' % class_names)

    # load background classes
    background_names = set(args.background)
    background_boxes = {}
    if args.background_file:
        _, ext = os.path.splitext(args.background_file)
        if '.json' == ext:
            with open(args.background_file) as json_file:
                background = json.load(json_file)
            background_names.update([n for n, b in background.items() if b is None])
            background_boxes.update({n: b for n, b in background.items() if b is not None})
        else:
            with open(args.background_file, 'r') as f:
                background_names.update(f.read().splitlines())

    logger.info('background class names: %s' % background_names)
    logger.info('background blind boxes: %s' % background_boxes)

    # define class colors
    colors = [(255, 0, 0)] * len(class_names)
    class_colors = {n: colors[i] for i, n in enumerate(class_names)}

    if args.no_recognition:
        recognizer = None
        min_rect_size = args.min_rect_size
    else:
        recognizer = RecognitionEngine(len(class_names), args.max_output_size, args.iou_threshold,
                                       args.min_box_conf, args.min_class_conf, args.min_contour_area,
                                       args.selector, args.weights_file)

    logger.info('recognizer model size is %s' % str(recognizer.model_size))

    if args.no_detection:
        detector = None
    else:
        detector = MovementDetector(args.min_contour_area, args.rectangle_separation,
                                    args.gray_threshold, args.gray_smoothing)

    if args.onvif_credentials is not None:
        onvif_connector = ONVIFConnector(urlparse(args.url).hostname, args.onvif_port,
                                         args.onvif_credentials[0], args.onvif_credentials[1], logger)
    else:
        onvif_connector = None

    processor = FrameProcessor(detector, recognizer, onvif_connector, logger, class_colors, class_names,
                               background_names, background_boxes, args.min_class_conf, args.background_overlap,
                               args.screenshot_dir, args.screenshot_quality, args.raw_screenshot_dir,
                               args.max_screen_size)

    # insert credentials into URL if required
    url = args.url.format(user=args.credentials[0],
                          passwd=args.credentials[1]) if args.credentials is not None else args.url

    if args.mjpg:
        watch_mjpg(url, args.credentials, args.retry_interval)
    else:
        watch(url, args.retry_interval)

    logger.info('Stop watching')
