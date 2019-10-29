# Daywatch
The project main goal is to detect and recognize moving objects using outdoor 
camera video stream. Pretrained [YOLOv3 model](https://pjreddie.com/darknet/yolo/) 
is used as recognizer. YOLOv3 is trained on COCO dataset. More details about COCO 
classes can be found [here](https://github.com/nightrome/cocostuff). 

When one or more object is detected, the screenshot can be saved. 
It is possible to manage background classes, objects of such classes 
do not trigger screenshot. The YOLOv3 part of the project is based on 
original [Kaggle notebook](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow).

If the camera supports [ONVIF](https://www.onvif.org/) protocol, it is possible to control camera
by press and release mouse left button on selected areas of the screen.
Both RTSP and MJPEG video streams are supported (for the latter use key `-mjpg`), 
which makes it possible to use cheap webcams plugged into router with 
OpenWRT and professional videosurveillance devices.

## Dependencies
* Python 3.6.x
* Tensorflow 1.1x
* OpenCV 4.x
* NumPy
* onvif_zeep

## Weights
Load YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights 
and use key `-wf` (or `--weights-file`) to provide path for weights file 
(default is `./yolov3.weights`)

## Getting started
Use key `-h` to read about all available options
```
daywatch.py -h
```
When focus on security feed window following hotkeys available:
* `<space>` manually save a screenshot
* `m` switch to multiscreen mode, which might be useful for tune motion detector parameters
* `b` show/hide background zones (when background objects are defined in json file)
* `c` in background mode: switch between background zones for specific classes
* `q` quit

## Remote control

If camera provides ONVIF endpoint and ONVIF credentials are provided  ( `-oc` or `--onvif-credentials` key), 
remote control is availabe based on continuous moves. If mouse right button is pressed 
at the left third part of the image, camera starts moving left, if it is pressed at the
right third part of the image, camera start moving right, and similar for the upper and
lower third parts of the image. These moves can be combined: if mouse right button is
pressed near the corner of the image, camera start moving top-left, top-right,
bottom-left or bottom-right, depending of the corner. Camera stops when mouse right button
is released. Due to non-zero latency, camera can stop moving with some delay. While 
camera is moving, no motion detection and object recognition is performed.
 
