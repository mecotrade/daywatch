import cv2
import urllib.request
import numpy as np
import base64
import time
import logging
import queue
import threading
from seaborn import color_palette
import yolov3
import argparse

from movement_detector import MovementDetector
from recognition_engine import RecognitionEngine
from frame_processor import FrameProcessor


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            while not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Queue.Empty:
                    break
            self.q.put(frame)

    def read(self):
        return self.q.get()


def get_authorization(credentials):
    base64string = base64.encodebytes(bytes(('%s:%s' % (credentials[0], credentials[1])), 'utf-8')).decode('utf-8').replace('\n','')
    return 'Basic %s' % base64string


def watch_mjpg(url, credentials):

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
            logger.info('Fail establish communication: %s' % ex)
            time.sleep(10)


def watch(url):

    cap = VideoCapture(url)

    loop = True
    while loop:

        frame = cap.read()
        loop = processor(frame)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', 
                       help='URL of the videostream')
    parser.add_argument('-c', '--credentials', nargs=2,
                       help='credential, username and password')
    parser.add_argument('-mrs', '--min-rect-size', type=int, nargs=2, default=yolov3.MODEL_SIZE,
                       help='minimal size of rectangle to be croped from initial frame for object recognition')
    parser.add_argument('-mca', '--min-contour-area', type=int, default=1000,
                        help='minimal area of the contour with detected motion')
    parser.add_argument('-s', '--sensetivity', type=int, default=16,
                        help='motion sensetivity: less is more sensetive')
    parser.add_argument('-rs', '--rectangle-separation', type=int, default=5,
                        help='mimimal distance bewteen rectangle edges for rectangles to be separated')
    parser.add_argument('-mos', '--max-output-size', type=int, default=10,
                        help='maximal possible number of detected object for each class')
    parser.add_argument('-ct', '--confidence-threshold', type=float, default=0.5,
                        help='when confidence is above this threshold object is considered to be classified')
    parser.add_argument('-iout', '--iou-threshold', type=float, default=0.5,
                        help='among all intersection boxes containing classified object those whose Intersection Over Union (iou) part is greater than this threshold are choosen')
    parser.add_argument('-wf', '--weights_file', default='yolov3.weights',
                        help='file with YOLOv3 weights. Must be downloaded from https://pjreddie.com/media/files/yolov3.weights')
    parser.add_argument('-nf', '--names-file', default='coco.names',
                        help='text file with COCO classes names, one name per line')
    parser.add_argument('-lf', '--log-file', default='watch.log',
                        help='log file')
    parser.add_argument('-sd', '--screenshot-dir',
                        help='directory where screenshots is stored, if not set, screenshots are not saved')
    parser.add_argument('-b', '--background', default=[], nargs='*',
                        help='names of background classes, objects of such classes do not trigger screenshot')
    parser.add_argument('-bf', '--background-file',
                        help='text file with background classes, objects of such classes do not trigger screenshot, one line per class, the order is not important')
    parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode')
    parser.add_argument('-m', '--mjpg', action='store_true', help='connect to MJPG stream source')
    
    args = parser.parse_args()

    # init logger
    handler = logging.FileHandler(args.log_file)
    handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    logger = logging.getLogger('watch')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.info('Start watching with following configuration:')
    for a, v in vars(args).items():
        # hide password in logs
        if 'credentials' == a and v:
            v = [v[0], '******']
        logger.info('  - %s: %s' % (a, v))

    # load COCO class names
    with open(args.names_file, 'r') as f:
        class_names = f.read().splitlines()
    logger.info('COCO class names: %s' % class_names)

    # load background classes
    background_names = set(args.background)
    if args.background_file:
        with open(args.background_file, 'r') as f:
            background_names.update(f.read().splitlines())
    logger.info('background class names: %s' % background_names)

    # define class colors
    colors = np.array(color_palette('hls', 80)) * 255
    class_colors = {n: colors[i] for i, n in enumerate(class_names)}

    detector = MovementDetector(args.min_contour_area, args.min_rect_size, args.rectangle_separation, args.sensetivity)
    recognizer = RecognitionEngine(class_names, args.max_output_size, args.iou_threshold, args.confidence_threshold, args.weights_file)
    processor = FrameProcessor(detector, recognizer, logger, class_colors, background_names, args.screenshot_dir)

    if args.mjpg:
        watch_mjpg(args.url, args.credentials)
    else:
        watch(args.url)

    logger.info('Stop watching')
