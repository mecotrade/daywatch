# daywatch
The project main goal is to detect and recognize moving objects using street camera raw stream. Pretrained [YOLOv3 model](https://pjreddie.com/darknet/yolo/) is used as recognizer. YOLOv3 is trained on COCO dataset. More details about COCO classes can be found [here](https://github.com/nightrome/cocostuff). 

When one or more object is detected, the screenshot can be saved. It is possible to manage background classes, objects of such classes do not trigger screenshot. The YOLOv3 part of the project is based on original [Kaggle notebook](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow).

## Dependencies
* Python 3.6.x
* Tensorflow 1.1x
* OpenCV 4.x
* NumPy
* onvif_zeep

## Weights
Load YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights and use key `-wf` (or `--weights-file`) to provide path for weights file (default is `./yolov3.weights`)

## Getting started
Use key `-h` to read about all available options
```
daywatch.py -h
```
When focus on security feed window following hotkeys available:
* `s` manually save a screenshot
* `m` switch to multiscreen mode, which might be useful for tune motion detector parameters
* `b` show/hide background zones (when background objects are defined in json file)
* `c` in background mode: switch between background zones for specific classes
* `q` quit

If camera provides ONVIF endpoint and ONVIF credentials are provided  ( `-oc` or `--onvif-credentials` key), 
the following remote control hotkeys are available:
* `w` move UP
* `a` move LEFT
* `d` move RIGHT
* `s` move DOWN  
