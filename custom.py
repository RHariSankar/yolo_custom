import tensorflow as tf
import sys
import cv2
import math
import numpy as np
from PIL import Image
from yolo_v3 import Yolo_v3
from utils import load_images, load_class_names, draw_boxes, draw_frame

tf.compat.v1.disable_eager_execution()

_MODEL_SIZE = (608, 608)
_CLASS_NAMES_FILE = './data/labels/obj.names'
_MAX_OUTPUT_SIZE = 100

iou_threshold = 0.3
confidence_threshold = 0.3
height_offset = 0.05
width_offset = 0.03
vertical_pieces = 2
horizontal_pieces = 2 

class_names = load_class_names(_CLASS_NAMES_FILE)
n_classes = len(class_names)

test_dir = "../../VisDrone2019-DET-val/images/"
test_img = test_dir+"0000271_06001_d_0000402.jpg"

image = tf.keras.preprocessing.image.load_img(test_img)   #Load image
image = tf.keras.preprocessing.image.img_to_array(image) 
shape = image.shape
height = shape[0]
width = shape[1]
offset_height = height_offset * height
offset_width = width_offset * width
vert_pieces = vertical_pieces
horiz_pieces = horizontal_pieces
crop_height = int(math.ceil(height/vert_pieces + offset_height))
crop_width = int(math.ceil(width/horiz_pieces + offset_width))

# Calculate crop coordinates
boxes = []
x1 = 0
y1 = 0
for i in range(vert_pieces):
    y = y1 + crop_height/height
    for j in range(horiz_pieces):
        x = x1 + crop_width/width
        if(y > 1):
            y = 1
        if(x > 1):
            x = 1
        boxes.append([y1,x1,y,x])
        x1 = x - (2*offset_width)/width
    x1 = 0
    y1 = y - (2*offset_height)/height
boxes = np.array(boxes)

images = []
images.append(image)
images = np.array(images)
num_boxes = boxes.shape[0]
box_indices = np.zeros(num_boxes)

crop_tensor = tf.image.crop_and_resize(images, boxes, box_indices, crop_size=[608,608], method='bilinear', extrapolation_value=0, name=None)
cropped_images = crop_tensor.eval(session=tf.compat.v1.Session())

batch_size = cropped_images.shape[0]

model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                max_output_size=_MAX_OUTPUT_SIZE,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)


inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, *_MODEL_SIZE, 3])
detections = model(inputs, training=False)
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))

with tf.compat.v1.Session() as sess:
    saver.restore(sess, './weights/model.ckpt')
    detection_result = sess.run(detections, feed_dict={inputs: cropped_images})

print(detection_result)                

#Aggregate detections