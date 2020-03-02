import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# image path
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1]
filename = dir_path +"/" + image_path
print(filename)
#image size and number of channels
image_size=128
num_channels=3
images = []

# read image
image = cv2.imread(filename)
image_out = image

# resize image so preprocessing is done exactly the same way as when training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)

#reshape the input of the network
x_batch = images.reshape(1, image_size,image_size,num_channels)

#restore the saved model
sess = tf.Session()

# recreate the network graph
saver = tf.train.import_meta_graph('BIC.meta')

#load the weights saved using
saver.restore(sess, tf.train.latest_checkpoint('./'))

#default graph which is restored
graph = tf.compat.v1.get_default_graph()

#y_pred is prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

#images are fed into placeholder
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, len(os.listdir(dir_path+"/training_set"))))

#feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)

# result is of this format [probabiliy_of_negative probability_of_positive]
#split the negative and positive probabilities from the result
result = (str(result).replace("[", ""))
result = (str(result).replace("]", ""))
negative = str(result).split()[0]
positive = str(result).split()[1]

#set final prediction
if (float(negative) > float(positive)):
    image_list_test_prediction = "negative"
else:
    image_list_test_prediction = "positive"

import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet(str(dir_path) + "/yolov3.weights", str(dir_path) + "/yolov3.cfg")
classes = []
with open(str(dir_path) + "/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = image_out

img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

cv2.putText(img, image_list_test_prediction, (0, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
cv2.imshow("Image", img)
cv2.waitKey(10000)