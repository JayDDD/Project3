
# Project 3

## Introduction

This is a project started by Udacity, the topic is Behavior Cloning. The training data come from the training driving by ourself, which will produce images from the cameras mounted on the car and a CVS file which contain basic information of the training driving process.

---

There are **2** python files: **model.py**, and **drive.py**, in which **drive.py** basically provided by Udacity.

## model.py
###There are several steps in the file.
1. Import all the data and resize all the images for only remaining the most useful information. There are some useless part in the     images which is not needed for training like skies. In addition, downsampling the images can effectively save the training time. Finally, I only remain one channel of the images mainly for saving training time. Details are in the python script. 
2. Set the images from three cameras as features and the turning angle as labels.

            **Features shape: (13557, 20, 64, 1)**
            **Labels shape: (13557,)**
3. Split the data into training and validation datasets, in which validation datasets are 20% of whole datasets.
4. Building model using Keras in reference to the tutorial by Udacity. Architecture details will be described later.
5. After model training, save the model and weights as **model.json** and **model.h5** as required. 


## drive.py
The main purpose for this python script is connect the code with the simulator using the model in **model.py**.



## Model Architecture

The following table is the Model Architecture table automatically generated using **model.summary()**.

            ____________________________________________________________________________________________________
            Layer (type)                     Output Shape          Param #     Connected to
            ====================================================================================================
            convolution2d_1 (Convolution2D)  (None, 18, 62, 16)    160         convolution2d_input_1[0][0]
            ____________________________________________________________________________________________________
            activation_1 (Activation)        (None, 18, 62, 16)    0           convolution2d_1[0][0]
            ____________________________________________________________________________________________________
            convolution2d_2 (Convolution2D)  (None, 16, 60, 8)     1160        activation_1[0][0]
            ____________________________________________________________________________________________________
            activation_2 (Activation)        (None, 16, 60, 8)     0           convolution2d_2[0][0]
            ____________________________________________________________________________________________________
            convolution2d_3 (Convolution2D)  (None, 14, 58, 4)     292         activation_2[0][0]
            ____________________________________________________________________________________________________
            activation_3 (Activation)        (None, 14, 58, 4)     0           convolution2d_3[0][0]
            ____________________________________________________________________________________________________
            convolution2d_4 (Convolution2D)  (None, 12, 56, 2)     74          activation_3[0][0]
            ____________________________________________________________________________________________________
            activation_4 (Activation)        (None, 12, 56, 2)     0           convolution2d_4[0][0]
            ____________________________________________________________________________________________________
            maxpooling2d_1 (MaxPooling2D)    (None, 6, 28, 2)      0           activation_4[0][0]
            ____________________________________________________________________________________________________
            dropout_1 (Dropout)              (None, 6, 28, 2)      0           maxpooling2d_1[0][0]
            ____________________________________________________________________________________________________
            flatten_1 (Flatten)              (None, 336)           0           dropout_1[0][0]
            ____________________________________________________________________________________________________
            dense_1 (Dense)                  (None, 16)            5392        flatten_1[0][0]
            ____________________________________________________________________________________________________
            activation_5 (Activation)        (None, 16)            0           dense_1[0][0]
            ____________________________________________________________________________________________________
            dropout_2 (Dropout)              (None, 16)            0           activation_5[0][0]
            ____________________________________________________________________________________________________
            dense_2 (Dense)                  (None, 1)             17          dropout_2[0][0]
            ====================================================================================================
            Total params: 7,095
            Trainable params: 7,095
            Non-trainable params: 0



```python

```
