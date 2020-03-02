#import necessary packages
import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import requests
import urllib
#set parent directory
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
