# Yolo v3 Object Detection in Tensorflow
Yolo v3 is an algorithm that uses deep convolutional neural networks to detect objects. <br> <br>
[Kaggle notebook](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow) 

## Getting started

### Prerequisites
This project is written in Python 3.6.6 using Tensorflow (deep learning), NumPy (numerical computing), Pillow (image processing), OpenCV (computer vision) and seaborn (visualization) packages.

```
pip install -r requirements.txt
```

### Downloading official pretrained weights
For Linux: Let's download official weights pretrained on COCO dataset. 

```
wget -P weights https://pjreddie.com/media/files/yolov3.weights
```
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and adding them to the weights folder.

### Using Custom trained weights
Add your custom weights file to weights folder and your custom .names file into data/labels folder.

Change 'n_classes=80' on line 97 of load_weights.py to 'n_classes=<number of classes in .names file>'.

Change './weights/yolov3.weights' on line 107 of load_weights.py to './weights/<custom weights file>'.

Change './data/labels/coco.names' on line 25 of detection.py to './data/labels/<custom names files>'.
  
### Save the weights in Tensorflow format
Save the weights using `load_weights.py` script.

```
python load_weights.py
```

## Running the model
Now you can run the model using `detect.py` script. Don't forget to set the IoU (Intersection over Union) and confidence thresholds.
### Usage
```
python detect.py <images/video> <iou threshold> <confidence threshold> <filenames>
```
### Images example
Let's run an example using sample images.
```
python detect.py images 0.5 0.5 data/images/dog.jpg data/images/office.jpg
```
Then you can find the detections in the `detections` folder.
<br>
You should see something like this.
```
detection_1.jpg
```
![alt text](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detection_1.jpg)
```
detection_2.jpg
```
![alt text](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detection_2.jpg)
### Video example
You can also run the script with video files.
```
python detect.py video 0.5 0.5 data/video/shinjuku.mp4
```
The detections will be saved as `detections.mp4` file.
![alt text](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detections.gif)

## To-Do List
* Model training

## Acknowledgments
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
* [A Tensorflow Slim implementation](https://github.com/mystic123/tensorflow-yolo-v3)
* [ResNet official implementation](https://github.com/tensorflow/models/tree/master/official/resnet)
* [DeviceHive video analysis repo](https://github.com/devicehive/devicehive-video-analysis)
* [A Street Walk in Shinjuku, Tokyo, Japan](https://www.youtube.com/watch?v=kZ7caIK4RXI)
