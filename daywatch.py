import cv2
import urllib.request
import numpy as np
import base64
import datetime
import time
import os
import tensorflow as tf
import logging
import inspect
from seaborn import color_palette
import yolov3
import argparse

from movement_detector import MovementDetector
from recognition_engine import RecognitionEngine


def get_authorization(credentials):
    base64string = base64.encodebytes(bytes(('%s:%s' % (credentials[0], credentials[1])), 'utf-8')).decode('utf-8').replace('\n','')
    return 'Basic %s' % base64string


def save_frame(screenshot_dir, frame_time, frame, prefix=''):

    today_screenshot_dir = os.path.join(screenshot_dir, frame_time.strftime('%Y%m%d'))
    if not os.path.exists(today_screenshot_dir):
        os.makedirs(today_screenshot_dir)
        print('Created screenshot directory %s' % today_screenshot_dir)

    cv2.imwrite(os.path.join(today_screenshot_dir, '%s%s.jpg' % (prefix, frame_time.strftime('%Y%m%d%H%M%S%f'))), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 100])


def start_watch(url, credentials, min_rect_size, class_names, background_names,
                min_contour_area=1000, sensetivity=16, rectangle_separation=5,
                max_output_size=10, confidence_threshold=0.5, iou_threshold=0.5,
                weights_file='yolov3.weights', log_file='watch.log', screenshot_dir='screenshots'):
    
    # init logger
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    logger = logging.getLogger('watch')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.info('Start watching with following configuration:')

    arguments, _, _, defaults = inspect.getargvalues(inspect.stack()[0][0])
    for a, v in defaults.items():
        if a in arguments:
            # hide password in logs
            if 'credentials' == a:
                v = [v[0], '******']
            logger.info('  - %s: %s' % (a, v))

    colors = np.array(color_palette('hls', 80)) * 255
    class_colors = {n: colors[i] for i, n in enumerate(class_names)}

    detector = MovementDetector(min_contour_area, min_rect_size, rectangle_separation, sensetivity)
    recognizer = RecognitionEngine(class_names, max_output_size, iou_threshold, confidence_threshold, weights_file)

    # start main loop
    loop = True
    while loop:

        try:
            request = urllib.request.Request(url)
            if credentials:
                request.add_header('Authorization', get_authorization(credentials=credentials))

            with urllib.request.urlopen(request, timeout=10) as stream:

                print('Communication established at %s' % datetime.datetime.now().strftime('%A %d %B %Y %H:%M:%S.%f')[:-3])

                data = bytearray()
                last_gray = None

                while True:

                    # current time
                    current_time = datetime.datetime.now()

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
                        screenshot = False

                        rects = detector(frame)
                        if rects:

                            # draw rectangles around subframes with movement
                            for x, y, w, h in rects:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            logger.debug('movement detected at %s' % rects)

                            # inspect subframes with movement
                            objects = recognizer(frame, rects)
                            for name, (x, y, w, h, conf) in objects.items():
                                cv2.rectangle(frame, (x, y), (x + w, y + h), class_colors[name], 1)
                                cv2.putText(frame, '%s %.1f%%' % (name, conf * 100), (x, y),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, class_colors[name], 1)
                            if len(set(objects.keys()) - background_names) > 0:
                                logger.info('[%s] %s' % (current_time.strftime('%Y%m%d%H%M%S%f'),
                                                         {n: r for n, r in objects.items()}))
                                screenshot = True

                        # text in the left top of the screen
                        cv2.putText(frame, 'Moving object detected' if len(rects) > 0 else 'All clear!', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                        # timestamp in the left bottom of the screen
                        cv2.putText(frame, current_time.strftime('%A %d %B %Y %H:%M:%S.%f')[:-3],
                                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                        # show the frame and record if the user presses a key
                        cv2.imshow('Security Feed', frame)
                            
                        key = cv2.waitKey(1) & 0xff

                        # if the 'q' key is pressed, break from the loop
                        if key == ord('q'):
                            loop = False
                            print('Exited')
                            break

                        # save frame if moving object detected or 's' key is pressed
                        if screenshot or key == ord('s'):                            
                            save_frame(screenshot_dir, current_time, frame)

            # close any open windows
            cv2.destroyAllWindows()

        except Exception as ex:
            print('Fail establish communication at %s: %s' % (datetime.datetime.now().strftime('%A %d %B %Y %H:%M:%S.%f')[:-3], ex))
            time.sleep(10)
            
    logger.info('Stop watching')


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
    parser.add_argument('-sf', '--screenshot-dir', default='screenshots',
                        help='directory where screenshots is stored')
    parser.add_argument('-b', '--background', default=[], nargs='*',
                        help='names of background classes, objects of such classes do not trigger screenshot')
    parser.add_argument('-bf', '--background-file',
                        help='text file with background classes, objects of such classes do not trigger screenshot, one line per class, the order is not important')
    
    args = parser.parse_args()

    # load COCO class names
    with open(args.names_file, 'r') as f:
        class_names = f.read().splitlines()

    # load background classes, if filename provided
    background_names = set(args.background)
    if args.background_file:
        with open(args.background_file, 'r') as f:
            background_names.update(f.read().splitlines())

    start_watch(url=args.url,
                credentials=args.credentials,
                min_rect_size=args.min_rect_size,
                class_names=class_names,
                background_names=background_names,
                min_contour_area=args.min_contour_area, 
                sensetivity=args.sensetivity, 
                rectangle_separation=args.rectangle_separation,
                max_output_size=args.max_output_size,
                confidence_threshold=args.confidence_threshold, 
                iou_threshold=args.iou_threshold,
                weights_file=args.weights_file,
                log_file=args.log_file,
                screenshot_dir=args.screenshot_dir)
