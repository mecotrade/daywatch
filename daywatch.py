import cv2
import urllib.request
import numpy as np
import base64
import datetime
import time
import locale
import os
import tensorflow as tf
import logging
import inspect
from seaborn import color_palette
import yolov3
import argparse

def get_authorization(credentials):
    base64string = base64.encodebytes(bytes(('%s:%s' % (credentials[0], credentials[1])), 'utf-8')).decode('utf-8').replace('\n','')
    return 'Basic %s' % base64string

def save_frame(image_folder, frame_time, frame, prefix=''):

    # image dir
    image_dir = os.path.join(image_folder, frame_time.strftime('%Y%m%d'))
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print('Created image folder %s' % image_dir)

    cv2.imwrite(os.path.join(image_dir, '%s%s.jpg' % (prefix, frame_time.strftime('%Y%m%d%H%M%S%f'))), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
def produce_rects(contours, min_contour_area, frame_size, min_size):
    
    rects = []
    for contour in contours:

        # process contour only if is is large enough
        if cv2.contourArea(contour) >= min_contour_area:
            
            # adjust bounding box of the contour to be at least as large as minimum size
            # detecting model input size is a good first guess for the minimum size
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_size[0]:
                x = min(max(0, x - (min_size[0] - w) // 2), frame_size[0] - min_size[0])
                w = min_size[0]
            if h < min_size[1]:
                y = min(max(0, y - (min_size[1]  - h) // 2), frame_size[1] - min_size[1])
                h = min_size[1]

            rects += [[x, y, w, h]]
        
    return rects

def merge_rects(rects, separation=0):
    
    new_merge = True
    while new_merge:
        new_rects = []
        new_merge = False
        for rect in rects:
            new_rects, merge = add_rect(new_rects, rect, separation=separation)
            if merge:
                new_merge = True
        rects = new_rects
    return rects
        
def intersect(a, b, separation=0):
    return not ((a[0] + a[2] <= b[0] + separation) or 
                (b[0] + b[2] <= a[0] + separation) or 
                (a[1] + a[3] <= b[1] + separation) or 
                (b[1] + b[3] <= a[1] + separation))

def add_rect(rects, r, separation=0):
    
    merge = False
    for i in range(len(rects)):
        if intersect(rects[i], r, separation):
            x, y, w, h = rects[i]
            rects[i][0] = min(r[0], x)
            rects[i][1] = min(r[1], y)
            rects[i][2] = max(r[0]+r[2], x+w) - rects[i][0]
            rects[i][3] = max(r[1]+r[3], y+h) - rects[i][1]
            merge = True
            break
    if not merge:
        rects += [r]
                
    return rects, merge

def start_watch(url, credentials, min_size, class_names, background_names,
                min_contour_area=1000, sensetivity=16, rectangle_separation=5,
                max_output_size=10, confidence_threshold=0.5, iou_threshold=0.5,
                weights_file='yolov3.weights', log_file='watch.log', screenshot_folder='screenshots'):
    
    # init logger
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    
    logger = logging.getLogger('watch')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    logger.info('Start watching with following configuration:')
    
    args, _, _, defaults = inspect.getargvalues(inspect.stack()[0][0])
    for a, v in  defaults.items():
        if a in args:
            # hide password in logs
            if 'credentials' == a:
                v = [v[0], '******']
            logger.info('  - %s: %s' % (a, v))
    
    # init YOLOv3
    colors = ((np.array(color_palette('hls', 80)) * 255))
    class_colors = {n: colors[i] for i, n in enumerate(class_names)}

    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
    model = yolov3.Yolo_v3(n_classes=len(class_names), model_size=yolov3._MODEL_SIZE,
                    max_output_size=max_output_size,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)
    outputs = model(inputs, training=False)

    model_vars = tf.global_variables(scope='yolo_v3_model')
    assign_ops = yolov3.load_weights(model_vars, weights_file)
    
    sess = tf.Session()
    sess.run(assign_ops)
        
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
                    if (len(new_bytes) == 0):
                        time.sleep(10)
                        break

                    data.extend(new_bytes)
                    begin_marker = data.find(b'\xff\xd8')
                    end_marker = data.find(b'\xff\xd9')

                    # new frame detected
                    if begin_marker!=-1 and end_marker!=-1:

                        jpg = data[begin_marker:end_marker+2]
                        data = data[end_marker+2:]

                        frame = cv2.imdecode(np.array(jpg), cv2.IMREAD_COLOR)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (21, 21), 0)

                        screenshot = False
                        if last_gray is not None:

                            # compute the absolute difference between the current frame and previous frame
                            frame_delta = cv2.absdiff(last_gray, gray)        
                            threshold = cv2.threshold(frame_delta, sensetivity, 255, cv2.THRESH_BINARY)[1]

                            # dilate the thresholded image to fill in holes, then find contours on thresholded image
                            threshold = cv2.dilate(threshold, None, iterations=2)
                            cnts, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            rects = merge_rects(produce_rects(cnts, min_contour_area, (frame.shape[1], frame.shape[0]), min_size), separation=rectangle_separation)
        
                            # detect objects in subframe
                            if rects:

                                inputs_value = [cv2.resize(frame[y:y+h,x:x+w,:], yolov3._MODEL_SIZE) for x, y, w, h in rects]
                                outputs_value = sess.run(outputs, feed_dict={inputs: inputs_value})
                                detections = yolov3.to_detections(outputs_value, class_names, confidence_threshold, iou_threshold)

                                # frame detected objects, if any
                                objects = {}
                                for (x, y, w, h), detection in zip(rects, detections):
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

                                    for name, boxes in detection.items():
                                        x_scale, y_scale = w/yolov3._MODEL_SIZE[0], h/yolov3._MODEL_SIZE[1]
                                        if name not in objects:
                                            objects[name] = []
                                        for box in boxes:
                                            x_obj = int(x+box[0]*x_scale)
                                            y_obj = int(y+box[1]*y_scale)
                                            w_obj = int(x+box[2]*x_scale) - x_obj
                                            h_obj = int(y+box[3]*y_scale) - y_obj
                                            cv2.rectangle(frame, (x_obj, y_obj), (x_obj+w_obj, y_obj+h_obj), class_colors[name], 1)
                                            cv2.putText(frame, '%s %.1f%%' % (name, box[4]*100), (x_obj, y_obj), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, class_colors[name], 1)
                                            objects[name] += [[x_obj, y_obj, w_obj, h_obj, box[4]]]

                                if len(set(objects.keys()) - background_names) > 0:
                                    logger.info('[%s] %s' % (current_time.strftime('%Y%m%d%H%M%S%f'), {n: r for n, r in objects.items()}))
                                    screenshot = True
            
                            # text in the left top of the screen
                            cv2.putText(frame, 'Moving object detected' if len(rects)>0 else 'All clear!', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                            # timestamp in the left bottom of the screen
                            cv2.putText(frame, current_time.strftime('%A %d %B %Y %H:%M:%S.%f')[:-3],
                                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                            # show the frame and record if the user presses a key
                            cv2.imshow('Security Feed', frame)
                            
                        last_gray = gray

                        key = cv2.waitKey(1) & 0xff

                        # if the 'q' key is pressed, break from the loop
                        if key == ord('q'):
                            loop = False
                            print('Exited')
                            break

                        # save frame if moving object detected or 's' key is pressed
                        if screenshot or key == ord('s'):                            
                            save_frame(screenshot_folder, current_time, frame)

            # close any open windows
            cv2.destroyAllWindows()

        except Exception as ex:
            print('Fail establish communication at %s: %s' % (datetime.datetime.now().strftime('%A %d %B %Y %H:%M:%S.%f')[:-3], ex))
            time.sleep(10)
            
    sess.close()
    logger.info('Stop watching')
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', 
                       help='URL of the videostream')
    parser.add_argument('-c', '--credentials', nargs=2,
                       help='credential, username and password')
    parser.add_argument('-mrs', '--min-rectangle-size', type=int, nargs=2, default=yolov3._MODEL_SIZE,
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
    parser.add_argument('-sf', '--screenshot-folder', default='screenshots',
                        help='folders for screenshots to be stored')
    parser.add_argument('-b', '--background', default=[], nargs='*',
                        help='names of background classes, objects of such classes do not trigger screenshot')
    parser.add_argument('-bf', '--background-file',
                        help='text file with background classes, objects of such classes do not trigger screenshot, one line per class, the order is not important')
    
    args = parser.parse_args()

    # load COCO class names
    with open(args.names_file, 'r') as f:
        class_names = f.read().splitlines()

    # load backgound classes, if filename provided
    background_names = set(args.background)
    if args.background_file:
        with open(args.background_file, 'r') as f:
            background_names.update(f.read().splitlines())

    start_watch(url=args.url,
                credentials=args.credentials,
                min_size=args.min_rectangle_size,
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
                screenshot_folder=args.screenshot_folder)