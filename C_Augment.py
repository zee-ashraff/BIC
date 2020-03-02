#import necessary packages
import os
import pandas as pd
import imageio
import imgaug.augmenters as iaa

#set parent directory
parent_directory=os.path.dirname(os.path.realpath(__file__))

#read in the train images dataframe
image_list_train = pd.read_csv(str(parent_directory) + "/" + "B_train_set.csv")
image_list_train= image_list_train[image_list_train.sentiment == "negative"]

for i in image_list_train.index:
    #if image is in highly positive sentiment class, create clones and add info to dataframe
        #one horizontally flipped
        #one vertically flipped
        #one with added noise
    image_path = str(image_list_train['filepath'][i])
    image = imageio.imread(image_path)

    #horizontal flip
    flip_hr = iaa.Fliplr(p=1.0)
    flip_hr_image = flip_hr.augment_image(image)
    flip_hr_path = parent_directory +"/training_set/negative/negative."+ str(i)+'H.jpg'
    imageio.imwrite(flip_hr_path, flip_hr_image)

     #vertical flip
    flip_vr = iaa.Flipud(p=1.0)
    flip_vr_image = flip_vr.augment_image(image)
    flip_vr_path = parent_directory +"/training_set/negative/negative."+ str(i)+'V.jpg'
    imageio.imwrite(flip_vr_path, flip_vr_image)

    #noise added
    gaussian_noise = iaa.AdditiveGaussianNoise(15, 20)
    noise_image = gaussian_noise.augment_image(image)
    noise_path = parent_directory +"/training_set/negative/negative."+ str(i)+'N.jpg'
    imageio.imwrite(noise_path, noise_image)
