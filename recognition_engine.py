import numpy as np
import tensorflow as tf
import cv2
import yolov3


class RecognitionEngine:

    def __init__(self, n_classes, max_output_size, iou_threshold, min_box_conf, min_class_conf, min_box_area,
                 selector, weights_file):

        self.iou_threshold = iou_threshold
        self.min_box_conf = min_box_conf
        self.min_class_conf = min_class_conf
        self.min_box_area = min_box_area

        if 'box' == selector:
            self.selector = lambda clusters, boxes: [cluster[np.argmax(boxes[cluster, 4])] for cluster in clusters]
        elif 'class' == selector:
            self.selector = lambda clusters, boxes: [cluster[np.argmax(np.max(boxes[cluster, 5:], axis=-1))]
                                                     for cluster in clusters]
        elif 'f1' == selector:
            def argmax_f1_score(cluster, boxes):
                max_class_conf = np.max(boxes[cluster, 5:], axis=-1)
                box_conf = boxes[cluster, 4]
                f1_score = 2 * max_class_conf * box_conf / (max_class_conf + box_conf)
                return np.argmax(f1_score)
            self.selector = lambda clusters, boxes: [cluster[argmax_f1_score(cluster, boxes)] for cluster in clusters]

        else:
            raise ValueError('Unknown selector %s' % selector)

        model = yolov3.Yolo_v3(n_classes=n_classes, max_output_size=max_output_size)
        self.model_size = model.model_size
        self.inputs = tf.placeholder(tf.float32, [None, self.model_size[0], self.model_size[1], 3])
        self.outputs = model(self.inputs, training=False)

        model_vars = tf.global_variables(scope='yolo_v3_model')
        assign_ops = yolov3.load_weights(model_vars, weights_file)

        self.sess = tf.Session()
        self.sess.run(assign_ops)

    def __del__(self):

        self.sess.close()

    def __call__(self, frame, rects):

        subframes = [cv2.resize(frame[y:y + h, x:x + w, :], self.model_size, interpolation=cv2.INTER_CUBIC)
                     for x, y, w, h in rects]
        # output values are [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes...]
        outputs_value = self.sess.run(self.outputs, feed_dict={self.inputs: subframes})
        detections = self.detect(outputs_value)

        objects = []
        for (x, y, w, h), boxes in zip(rects, detections):
            x_scale, y_scale = w / self.model_size[0], h / self.model_size[1]
            for box in boxes:
                x_obj = int(x + box[0] * x_scale)
                y_obj = int(y + box[1] * y_scale)
                w_obj = int(x + box[2] * x_scale) - x_obj
                h_obj = int(y + box[3] * y_scale) - y_obj
                objects += [[x_obj, y_obj, w_obj, h_obj] + box[4:]]

        return objects

    # https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
    # https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Malisiewicz et al.
    def clusterize(self, boxes):

        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        clusters = dict()
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            clusters.update({i: [i]})

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # indices of having overlapping with i-th box
            cluster = np.nonzero(overlap > self.iou_threshold)[0]
            clusters[i] += idxs[cluster].tolist()

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate([[last], cluster]))

        return list(clusters.values())

    def detect(self, outputs):
        """
        :param outputs:
                the raw output of the model, list of arrays
                [x1, y1, x2, y2, conf, class1, class2, ...]
                where (x1, y1) is upper-right coordinate of the box,
                (x2, y2) is lower-left coordinate of the box,
                conf is the confidence of the box, and
                class1, class2, ... are confidence of the classes
        :return:
                list (one item per subframe) of list (one box per object)
                of [x1, y1, w, h, conf, class1, class2, ...],
                where (x1, y1) is upper-left box coordinate, w is the box width,
                h is the box height, conf is the box confidence, and
                class1, class2, ... are class confidences
        """
        boxes_dicts = []

        detections = []
        for boxes in outputs:

            boxes = boxes[boxes[:, 4] > self.min_box_conf]
            boxes = boxes[boxes[:, 2] * boxes[:, 3] > self.min_box_area]
            if len(boxes) > 0:
                clusters = self.clusterize(boxes[:, :4])

                mean_class_conf = np.array([np.average(boxes[cluster][:, 5:], weights=boxes[cluster][:, 4], axis=0)
                                            for cluster in clusters])
                boxes = boxes[self.selector(clusters, boxes), :]
                boxes[:, 5:] = mean_class_conf

                boxes = boxes[np.max(mean_class_conf, axis=1) > self.min_class_conf]

                detections += [boxes.tolist()]

        return detections

