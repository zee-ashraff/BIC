#import necessary packages
import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

#set parent directory using the os package
parent_directory=os.path.dirname(os.path.realpath(__file__))

#create subdirectories where the images will be saved
path = os.path.join(parent_directory, "training_set/positive")
os.makedirs(path, exist_ok = True)
path = os.path.join(parent_directory, "training_set/negative")
os.makedirs(path, exist_ok = True)
path = os.path.join(parent_directory, "test_set/positive")
os.makedirs(path, exist_ok = True)
path = os.path.join(parent_directory, "test_set/negative")
os.makedirs(path, exist_ok = True)

#rename columns and only select the sentiment and URL
image_list = pd.read_csv(str(parent_directory) + "/" + "A_full_set.csv")
image_list.rename(columns = {'which_of_these_sentiment_scores_does_the_above_image_fit_into_best':'sentiment', 'imageurl':'url'}, inplace = True)
image_list = image_list[['sentiment', 'url']]

#initialize new column which tracks whether the image exists or not
image_list['exists']="0"

#loop through entire dataframe
for i in image_list.index:
    #select image url for each record and get response
    url = image_list['url'][i]
    resp = requests.get(url)

    #mark if response code is 200 or not, indicating if image exists or not
    if resp.status_code==200:
        image_list['exists'][i]=1
    else:
        image_list['exists'][i]=-1

#filter out records where images do not exist, and drop unnecessary columns
image_list = image_list[image_list.exists == 1]
image_list = image_list[['sentiment', 'url']]

#select features for X and y
X = image_list['url']
y = image_list['sentiment']

#split all into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#concatenate X and Y to get complete train and test sets
image_list_train = pd.concat([X_train, y_train], axis=1)
image_list_test = pd.concat([X_test, y_test], axis=1)

#output CSVs files in parent directory with list of images in the train and test sets
image_list_train.to_csv(str(parent_directory) + "/" + 'A_train_set.csv', index = False)
image_list_test.to_csv(str(parent_directory) + "/" + 'A_test_set.csv', index = False)
