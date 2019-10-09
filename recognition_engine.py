import numpy as np
import tensorflow as tf
import cv2
import yolov3


class RecognitionEngine:

    def __init__(self, class_names, max_output_size, iou_threshold, confidence_threshold, weights_file):

        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.model_size = yolov3._MODEL_SIZE

        self.inputs = tf.placeholder(tf.float32, [None, self.model_size[0], self.model_size[0], 3])
        model = yolov3.Yolo_v3(n_classes=len(self.class_names), model_size=self.model_size, max_output_size=max_output_size)
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
        detections = self.detect2(outputs_value, self.class_names)

        # reshape detections back to original frame
        objects = {}
        for (x, y, w, h), detection in zip(rects, detections):
            x_scale, y_scale = w / self.model_size[0], h / self.model_size[1]
            for name, boxes in detection.items():
                if name not in objects:
                    objects[name] = []
                for box in boxes:
                    x_obj = int(x + box[0] * x_scale)
                    y_obj = int(y + box[1] * y_scale)
                    w_obj = int(x + box[2] * x_scale) - x_obj
                    h_obj = int(y + box[3] * y_scale) - y_obj
                    objects[name] += [[x_obj, y_obj, w_obj, h_obj, box[4]]]

        return objects

    # Malisiewicz et al.
    def non_max_suppression_fast(self, boxes):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

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
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

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

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.iou_threshold)[0])))

        return pick

    def detect(self, outputs, class_names):
        boxes_dicts = []

        for boxes in outputs:
            boxes = boxes[boxes[:, 4] > self.confidence_threshold]
            classes = np.argmax(boxes[:, 5:], axis=-1)
            classes = np.expand_dims(classes, axis=-1)
            boxes = np.concatenate([boxes[:, :5], classes], axis=-1)

            boxes_dict = dict()
            for cls in range(len(class_names)):
                mask = np.reshape(boxes[:, 5] == cls, [-1])
                if len(mask) != 0:
                    class_boxes = boxes[mask, :]
                    boxes_coords = class_boxes[:, :4]
                    indices = self.non_max_suppression_fast(boxes_coords)
                    class_boxes = class_boxes[indices]
                    if len(class_boxes) > 0:
                        boxes_dict[class_names[cls]] = class_boxes[:, :5]
            boxes_dicts.append(boxes_dict)

        return boxes_dicts

    def detect2(self, outputs, class_names):
        boxes_dicts = []

        for boxes in outputs:

            boxes = boxes[boxes[:, 4] > self.confidence_threshold]
            indices = self.non_max_suppression_fast(boxes[:, :4])
            boxes = boxes[indices]

            boxes_dict = dict()
            boxes_classes = [(class_names[cls], box[:5].tolist()) for cls, box in zip(np.argmax(boxes[:, 5:], axis=-1), boxes)]
            [boxes_dict[name].append(box) if name in list(boxes_dict.keys()) else boxes_dict.update({name: [box]}) for name, box in boxes_classes]

            boxes_dicts.append(boxes_dict)

        return boxes_dicts
