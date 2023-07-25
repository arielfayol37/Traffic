# Traffic Sign Recognition with TensorFlow

## Background

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, we used TensorFlow to build a neural network to classify road signs based on an image of those signs. We used the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

## Understanding

We have a dataset with labeled road sign images, and we need to implement the following functions in traffic.py.

### `load_data(data_dir)`

The `load_data` function takes the data directory path as an argument and returns image arrays and labels for each image in the data set. We assume that the data directory contains 43 subdirectories, one for each category (numbered 0 through NUM_CATEGORIES - 1). Inside each category directory, there are several image files. We use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray, resize them to have width IMG_WIDTH and height IMG_HEIGHT, and return a tuple (images, labels) where images is a list of numpy.ndarrays representing the images and labels is a list of integers representing the category number for each image.

### `get_model()`

The `get_model` function returns a compiled neural network model. We assume that the input to the neural network is of shape (IMG_WIDTH, IMG_HEIGHT, 3) representing an image with three values for each pixel (red, green, and blue). The output layer of the neural network has NUM_CATEGORIES units, one for each of the traffic sign categories. The architecture of the neural network is up to us to decide. We may experiment with different numbers of convolutional and pooling layers, different filter sizes for convolutional layers, different pool sizes for pooling layers, different numbers and sizes of hidden layers, and dropout.
