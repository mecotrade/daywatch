# daywatch
The project main goal is to detect and recognize moving objects using street camera raw stream. Pretrained [YOLOv3 model](https://pjreddie.com/darknet/yolo/) is used as recognizer. YOLOv3 is trained on COCO dataset. More details about COCO classes can be found [here](https://github.com/nightrome/cocostuff). 

When one or more object is detected, the screenshot can be saved. It is possible to manage background classes, objects of such classes do not trigger screenshot. The YOLOv3 part of the project is based on original [Kaggle notebook](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow).

## Dependencies
* Python 3.6.x
* Tensorflow 1.1x
* OpenCV 4.x
* NumPy
* Seaborn

## Weights
Load YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights and use key `-w` to provide path for weights file (default is `./yolov3.weights`)

## Getting started
Use key `-h` to read about all available options
```
daywatch.py -h
```
When focus on security feed window press `s` to manually save a screenshot, press `q` to quit the program.
