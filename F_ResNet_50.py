#import necessary packages
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define and move to dataset director
parent_directory=os.path.dirname(os.path.realpath(__file__))
datasetdir = parent_directory + "/" + "training_set"
os.chdir(datasetdir)
batch_size = 30

def generators(shape, preprocessing):
    #generate validation and train sets
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True,
        validation_split = 0.1,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width),
        classes = ('positive','negative'),
        batch_size = batch_size,
        subset = 'training',
    )

    val_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width),
        classes = ('positive','negative'),
        batch_size = batch_size,
        subset = 'validation'
    )
    return train_dataset, val_dataset


def plot_history(history, yrange):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)

    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    plt.show()

resnet50 = keras.applications.resnet50
train_dataset, val_dataset = generators((224,224), preprocessing=resnet50.preprocess_input)

conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in conv_model.layers:
    layer.trainable = False
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
full_model.summary()

full_model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.001),
                  metrics=['acc'])
history = full_model.fit_generator(
    train_dataset,
    validation_data = val_dataset,
    workers=10,
    epochs=1,
)

plot_history(history, yrange=(0.9,1))

full_model.save_weights('resnet50.h5')

conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in conv_model.layers:
    layer.trainable = False
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)

full_model.load_weights('resnet50.h5')

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import pandas as pd

#set parent directory
parent_directory=os.path.dirname(os.path.realpath(__file__))
# First, pass the path of the image
image_list_test = pd.read_csv(str(parent_directory) + "/" + "B_test_set.csv")
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

for i in (image_list_test.index):
    filename = image_list_test['filepath'][i]
    img_path = filename
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(full_model.predict(x))
    result = str(full_model.predict(x))

    print(result)

    result = (result.replace("[", ""))
    result = (result.replace("]", ""))

    positive = result.split()[0]
    negative = result.split()[1]


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
    #confusion matrix output
    print("TP:" + str(true_positive) + ", FP:" + str(false_positive) + ", FN:" + str(false_negative) +  ", TN:" + str(true_negative))
