#import necessary packages
import os
import pandas as pd
import requests

#set parent directory
parent_directory=os.path.dirname(os.path.realpath(__file__))

#read in the train images dataframe
image_list_train = pd.read_csv(str(parent_directory) + "/" + "A_train_set.csv")
#create two new columns
image_list_train['filepath']="0"
image_list_train['filename']="0"

#for loop to loop through entire train dataframe and download the images placing them in appropriate folders as .jpg
for i in image_list_train.index:
    #extract URL
    url = image_list_train['url'][i]
    #check sentiment, determine path, download file to train folder and populate filename and filepath
    if (image_list_train['sentiment'][i]=='Highly positive')or(image_list_train['sentiment'][i]=='Positive')or(image_list_train['sentiment'][i]=='Neutral'):
        path = os.path.join(parent_directory, "training_set/positive")
        image = requests.get(url).content
        file = path.replace('\\', '/')+"/positive."+str(i)+".jpg"
        with open(file, 'wb') as handler:
            handler.write(image)
        image_list_train['sentiment'][i] = "positive"
        image_list_train['filepath'][i]=file
        image_list_train['filename'][i]="positive."+str(i)+".jpg"
    #check sentiment, determine path, download file to train folder and populate filename and filepath
    else:
        path = os.path.join(parent_directory, "training_set/negative")
        image = requests.get(url).content
        file = path.replace('\\', '/')+"/negative."+str(i)+".jpg"
        with open(file, 'wb') as handler:
            handler.write(image)
        image_list_train['sentiment'][i] = "negative"
        image_list_train['filepath'][i]=file
        image_list_train['filename'][i]="negative."+str(i)+".jpg"

#select relevant columns from dataframe and generate train csv
image_list_train = image_list_train[['filename', 'sentiment', 'filepath']]
image_list_train.to_csv(str(parent_directory) + "/" + 'B_train_set.csv', index = False)


#read in the test images dataframe
image_list_test = pd.read_csv(str(parent_directory) + "/" + "A_test_set.csv")
#create two new columns
image_list_test['filepath']="0"
image_list_test['filename']="0"

#for loop to loop through entire test dataframe and download the images placing them in appropriate folders
for i in image_list_test.index:
    #extract URL
    url = image_list_test['url'][i]
    #check sentiment, determine path, download file to test folder and populate filename and filepath
    if (image_list_test['sentiment'][i]=='Highly positive')or(image_list_test['sentiment'][i]=='Positive')or(image_list_test['sentiment'][i]=='Neutral'):
        path = os.path.join(parent_directory, "test_set/positive")
        image = requests.get(url).content
        file = path.replace('\\', '/')+"/positive."+str(i)+".jpg"
        with open(file, 'wb') as handler:
            handler.write(image)
        image_list_test['sentiment'][i] = "positive"
        image_list_test['filepath'][i]=file
        image_list_test['filename'][i]="positive."+str(i)+".jpg"
    #check sentiment, determine path, download file to test folder and populate filename and filepath
    else:
        path = os.path.join(parent_directory, "test_set/negative")
        image = requests.get(url).content
        file = path.replace('\\', '/')+"/negative."+str(i)+".jpg"
        with open(file, 'wb') as handler:
            handler.write(image)
        image_list_test['sentiment'][i] = "negative"
        image_list_test['filepath'][i]=file
        image_list_test['filename'][i]="negative."+str(i)+".jpg"

#select relevant columns from dataframe and generate train csv
image_list_test = image_list_test[['filename', 'sentiment', 'filepath']]
image_list_test.to_csv(str(parent_directory) + "/" + 'B_test_set.csv', index = False)
