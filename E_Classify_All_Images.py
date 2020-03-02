import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import os
import pandas as pd

#set parent directory
parent_directory=os.path.dirname(os.path.realpath(__file__))
# First, pass the path of the image
image_list_test = pd.read_csv(str(parent_directory) + "/" + "0004_Download_test.csv")
image_list_test.rename(columns = {'sentiment':'truth'}, inplace = True)
image_list_test['prediction']="0"
image_list_test['true_positive']=0
image_list_test['false_positive']=0
image_list_test['true_negative']=0
image_list_test['false_negative']=0

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

#for i in image_list_test.index:
for i in (image_list_test.index):
    filename = image_list_test['filepath'][i]
    image_size=128
    num_channels=3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during test
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('BIC-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")
    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir('C:/Users/zeesh/Desktop/Broadview_Image_Classifier/test_set'))))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=str(sess.run(y_pred, feed_dict=feed_dict_testing))
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    result = (result.replace("[", ""))
    result = (result.replace("]", ""))

    negative = result.split()[0]
    positive = result.split()[1]

    if (float(negative) > float(positive)):

        image_list_test_prediction = "negative"
    else:
        image_list_test_prediction = "positive"
    print ()
    if (image_list_test['truth'][i] == "positive" and image_list_test_prediction == "positive"):
        true_positive = true_positive + 1
    elif (image_list_test['truth'][i] == "positive" and image_list_test_prediction == "negative"):
        false_negative = false_negative + 1
    elif (image_list_test['truth'][i] == "negative" and image_list_test_prediction == "negative"):
        true_negative = true_negative + 1
    elif (image_list_test['truth'][i] == "negative" and image_list_test_prediction == "positive"):
        false_positive = false_positive + 1
    print(image_list_test['filepath'][i] + ": " + image_list_test_prediction + " - TP:" + str(
        true_positive) + ", FP:" + str(false_positive) + ", TN:" + str(true_negative) + ", FN:" + str(false_negative))
