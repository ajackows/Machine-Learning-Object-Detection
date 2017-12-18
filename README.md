# Machine-Learning-Object-Detection
This repository is a program that uses the Tensor flow API to train a neural network to be able to detect an object in real time
In its current state the program will detect and locate the NASA logo from a webcam stream.

This project retrains a premade model to detect a completely new object. This uses the Single Shot Multibox Detector (SSD) model with mobienet, this was selected due to being a lightweight model allowing the program to run in realtime, there is stil significant dealy and a decent computer might still be reqired.

This project makes use of the Tensor Flow Models Library Found here and is based off some of the research examples:

https://github.com/tensorflow/models

##Requirements to run project 
(note as of writing this has only been tested on Ubuntu 16.04, results may vary from operating systems)

Python 3.6 with Anaconda
Protobuf V 3.5
Tensor Flow V1.4.0 or above
Open CV V3.0.0 **NOTE: new versions of OpenCV will not work as they use an older version of Protobuf that will result in a compilation error**

More notes on how to install some these frameworks will be available in the accompanied report

##Running Project:

Once all the necessary Libaries and frameworks are installed:

Clone Directory

Navigate in terminal to Directory above the cloned directory (if you type 'ls' this directory should appear in your lists)
rename the directory to Object-Detection (this is so you can run the follow script without error)

It may be necessary to run the following comand in terminal
protoc object-detection/protos/*.proto --python_out=.

Navigate into this directory (cd object-detection) Note the changed name

run command:
python3 imgRec.py


##To train your own data:

If you wish to train your own image detection, then you will need to clone the tensor flow model library linked above.
In the research folder is a object detection subdirectory, that is where most of the framework of this project was derived from.

First you will need a large selection of images with your desired objects labeled (100 is the bare minumin this project used approxiamtley 130 images, but the more the better)

For this project Labelimg was used for the label imaging process found here:https://github.com/tzutalin/labelImg
This program will generat XML files that will define where out object is in the training image.

Next the generated xml files will need to be converted to CSV files, this will assist in the generation of Tensor flow records, the files that the Tensor flow training program will use to train or model correctly.

Make sure to have the following file structure with the images and csv files:

Top-directory
-data/
  -train_labels.csv
  -test_labels.csv
-images/
  -train/
    -training images (preferably jpeg)
  -test/
    -testing images (preferably jpeg)
-training

Now with this file structure you can run the following program from the top directory to generate the tensor flow records:
https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

this will generate tf record files for the training and test data, note it may be necessary to move the images into the image directory instead of within the train or test images directory.

Note: the program will need to be modified to account for your model
In the code there is a TODO where you change the labels and ID's for the images to be trained shown below:

TO-DO replace this with label map

def class_text_to_int(row_label):
    if row_label == 'NASA': <-- custom change that I did , you can add more image labels if you want
        return 1
    else:
        None

From the top directory run:

python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

Once the training data is configured move the images, data, and training folders into the object detection sub-directory in the tensor flow models directory (the one linked earlier):

:Models/research/object_detection

Next we will obtain the model from the tensor flow github:
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

This is the same model used in the examples of tensor flow object detection. We will then need to obtain the configuration file for this model from the samples directory within the object detection directory

models-> research -> obeject-detection -> samples -> config -> ssd_mobilenet_v1.config

The path and needs image classifiers need to be modified to fit the the images you wish to detect. The config file will also posess other parameters to the trianing record and test record you may with to edit.
This config determines how the nerual network will be configured, how many epochs, how many evaluations, and how many classifiers. Add this edited file to your training subdirectory within the object-detection directory.
once this is done in the training subdirectory add a  file: object-detection.pbtxt
Write:
This will change with your object label, a copy of the training forlder used here is in this repository
item {
  id: 1 
  name: 'NASA'
}

you may need to execute the following command from the "research" directory
- models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Finally we train the model, this may take some time depending on your computer, natually GPU based machines will run muhc faster

python3 train.py --logtostderr --train_dir=training --pipeline_config_path=training/ssd_mobilenet_v1_coco.config

We train until we get a low enough error to be satisfactory.

we will see that in the training folder we will see checkpoint files. Thses are to be used to generate our trained model

If you have closed your terminal since you may need to rerun the following command in the research directory:
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Then from the tensor flow object-detection directory run the export_inference_graph program:
python3 export_inference_graph.py input_type image_tensor pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-5621 \
    --output_directory NASA_inference_graph
(the inference graph name will be different depending on what you named you dataset, we also will have a different checkpoint depending on the point you stopped traning you data, for this repository I trained for 5621 steps)

**Note:** make sure you have three different checkpoint files for the step you choose, otherwise you'll get an error.

Finally you reaplce all the relevant files in the Machine Learning Project folder and modify "imgRec.py" to look for your model 

MODEL_NAME ='NASA_inference_graph'

you can also modify the Jupyter notebook within the tensor flow object-detection directory to test your model on still images,  you will need to add iamges in the "test-images" folder in the directory.
