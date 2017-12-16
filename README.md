# Machine-Learning-Object-Detection
This repository is a program that uses the Tensor flow API to train a neural network to be able to detect an object in real time

This project makes use of the Tensor Flow Models Library Found here:

https://github.com/tensorflow/models

Requirements to run project (note as of writing this has only been tested on Ubuntu 16.04, results may vary from operating systems.

Python 3.6 with Anaconda
Protobuf V 3.5
Tensor Flow V1.4.0 or above
Open CV V3.0.0 NOTE: new versions of OpenCV will not work as they use an older version of Protobuf that will result in a compilation error

Running Project:

Once all the necessary Libaries and frameworks are installed:

Clone Directory
protoc object_detection/protos/*.proto --python_out=.
