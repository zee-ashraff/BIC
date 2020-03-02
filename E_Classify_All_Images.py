import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import os
import pandas as pd
import re


#set parent directory
parent_directory=os.path.dirname(os.path.realpath(__file__))

#read in dataframe containing list of test images, rename columns, and create
image_list_test = pd.read_csv(str(parent_directory) + "/" + "B_test_set.csv")
image_list_test.rename(columns = {'sentiment':'truth'}, inplace = True)

#initialize confusion matrix counters
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

#read through entire list and process images one by one
for i in (image_list_test.index):

    #get filepath, set image size, number of channels, and images[]
    filename = image_list_test['filepath'][i]
    image_size=128
    num_channels=3
    images = []

    # Read the image
    image = cv2.imread(filename)

    # Resize  image
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    #reshape input to network
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    #restore the saved model
    sess = tf.Session()

    #recreate network graph
    saver = tf.train.import_meta_graph('BIC.meta')

    #load  weights saved using the restore method
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    #access the default graph which we have restored
    graph = tf.get_default_graph()

    #y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    #images fed to input placeholder
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir(str(parent_directory)+'/test_set'))))


    #feed_dict required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=str(sess.run(y_pred, feed_dict=feed_dict_testing))

    # result is of this format [probabiliy_of_negative probability_of_positive]
    #split the negative and positive from the results
    result=(str(result).replace("[", ""))
    result = (str(result).replace("]", ""))
    negative = result.split()[0]
    positive = result.split()[1]

    #classify image depending on the values from above
    if(float(negative)>float(positive)):

        image_list_test_prediction ="negative"
    else:
        image_list_test_prediction ="positive"

    #confusion matrix counters
    if (image_list_test['truth'][i]=="positive" and image_list_test_prediction=="positive"):
        true_positive = true_positive + 1
    elif (image_list_test['truth'][i]=="positive" and image_list_test_prediction=="negative"):
        false_negative = false_negative + 1
    elif (image_list_test['truth'][i]=="negative" and image_list_test_prediction=="negative"):
        true_negative = true_negative + 1
    elif (image_list_test['truth'][i]=="negative" and image_list_test_prediction=="positive"):
        false_positive = false_positive + 1
    #confusion matrix output
    print("TP:" + str(true_positive) + ", FP:" + str(false_positive) + ", FN:" + str(false_negative) +  ", TN:" + str(true_negative))