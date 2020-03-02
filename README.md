# BIC
High Level Description of Code
There are eight different .py files associated with BIC’s setup. A_Filter_and_Split.py’s first step is creating the directories where the images will be downloaded to. Next, using A_full_set.csv, it sends requests to the image URLs to determine which images are unavailable, and filters them out. Finally, it splits the filtered image list into train and test sets, at a ratio of 80:20. This code generates two CSVs containing the full list of images in the test and train directories. B_Download.py is used to download the images into negative and positive directories, for both train and test sets. The output of this code is two CSV files with the file path of each local image. C_Augment.py removes the class imbalance by creating augmented copies of the negative train set.

D_Dataset.py is used to define functions which return the train dataset and validation dataset, along with several data attributes. D_Train.py calls D_Dataset.py while execution. D_Train.py is the code which trains the network. First, the structure of the convolutional layers is defined, along with the other layers’ specifics. The session is then created and the images in the training dataset are passed in to train the network. Once training is complete the .py files with the prefix “E_” are used to perform various forms of classifications. E_Classify_Single_Image.py is used to classify images one at a time. The output of this code is the input image with a textbox containing the class of the image. E_Classify_and_Detect.py is an extension of E_Classify_Single_Image.py, as it classifies images in addition to detecting objects. E_Classify_All_Images.py classifies the entire test sets and outputs the confusion matrix values. The files F_VGG16.py, F_VGG19.py and F_ResNet_50.py are used to execute transfer learning.

Required Packages
The packages listed below must be set up in the Anaconda environment before the code can be executed.

Package	                        Install Command
re	                            conda install -c anaconda regex
pandas                          conda install pandas
requests	                      conda install -c anaconda requests
sklearn	                        conda install scikit-learn
os	                            no need to install, only needs to be imported
imageio	                        conda install -c conda-forge imageio
imgaug	                        conda install imgaug
cv2	                            conda install -c conda-forge opencv
glob	                          conda install -c conda-forge glob2
numpy	                          conda install -c conda-forge numpy
tensorflow	                    conda install -c conda-forge tensorflow
matplotlib	                    conda install -c conda-forge matplotlib
keras (for transfer learning) 	conda install -c conda-forge keras
Table 2: Require packages

Implementation Instructions
1.	Create a new folder called “BIC” on your machine
2.	Download and place the following files in the newly created folder:
    a.	A_Filter_and_Split.py	            b.	A_full_set.csv
    c.	B_Download.py	                    d.	C_Augment.py
    e.	D_Dataset.py	                    f.	D_Train.py
    g.	E_Classify_All_Images.py	        h.	E_Classify_and_Detect.py
    i.	E_Classify_Single_Image.py	      j.	F_ResNet_50.py
    k.	F_VGG16.py	                      l.	F_VGG19.py
3.	Open Anaconda and activate tfenv
4.	Run A_Filter_and_Split.py [2 hour run time] and ensure that the following directories and files have been created:
    a.	BIC/Training_set	                b.	BIC/Training_set/negative 
    c.	BIC/Training_set/positive	        d.	BIC/Test_set
    e.	BIC/Test_set/negative             f.	BIC/A_test_set.csv
    g.	BIC/Test_set/positive             h.	BIC/A_train_set.csv
5.	Run B_Download.py [2 hour run time] and ensure that the folders in listed in 4.b, 4.c, 4.e and 4.f have been populated with images, and BIC/B_test_set.csv and BIC/B_train_set.csv have been created
6.	Run C_Augment.py, and confirm that augmented versions of the negative train images have been created
7.	Run D_Train.py, which may run for several hours depending on the num_iteration parameter’s value, which controls the number of epochs
8.	Run E_Classify_All_Images.py to classify the entire dataset and generate running confusion matrix values [5 hour run time], or skip to step 9 if single image classification is desired
9.	Run E_Classify_Single_Image.py by specifying the image path in the following manner, for example for negative.0.jpg:
        python C:\Users\zeesh\Desktop\BIC\E_Classify_Single_Image.py test_set\negative\negative.0.jpg
10.	Run F_ResNet_50.py for transfer learning results using ResNet-50 [5 hour execution time]
11.	Run F_VGG16.py for transfer learning results using VGG16.py [5 hour execution time]
12.	Run F_VGG19.py for transfer learning results using VGG19.py [5 hour execution time]

Optional Object Detection Implementation Instructions
1.	To run the optional object detection algorithm, download the files listed below:
a.	coco.names	b.	yolov3.cfg  c.	yolov3.weights
2.	Run E_Classify_and_Detect.py by specifying the image path in the following manner:
      python C:\Users\zeesh\Desktop\BIC\E_Classify_and_Detect.py test_set\negative\negative.0.jpg
