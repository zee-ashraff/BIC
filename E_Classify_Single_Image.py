import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# image path
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1]
filename = dir_path +"/" + image_path
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

#output image with final prediction
cv2.putText(image_out, image_list_test_prediction, (0, 40), cv2.FONT_HERSHEY_PLAIN, 3,(0,0,0) , 3)
cv2.imshow("Image", image_out)
cv2.waitKey(10000)