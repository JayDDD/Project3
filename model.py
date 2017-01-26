import argparse
import os
import sys
import csv
import base64
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
import h5py
import pickle
import math
import tensorflow as tf
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K



file = "driving_log.csv"
data = []
with open(file) as F:
    content = csv.reader(F)
    next(content)
    for i in content:
        data.append(i) 

features = ()
labels = ()

# Resize the image and turn them into list
def load_image(data_line, j):
    img = plt.imread(data_line[j].strip())[55:155:5,0:-1:5,0]
    lists = img.flatten().tolist()
    return lists

# For each item in data, convert camera images to single list
# and save them into features list.
for i in tqdm(range(int(len(data))), unit='items'):
    for j in range(3):
        features += (load_image(data[i],j),)

item_num = len(features)
print("features size", item_num)

# A single list will be convert back to the original image shapes.
# Each list contains 3 images so the shape of the result will be
# 54 x 80 where 3 images aligned vertically.
features = np.array(features).reshape(item_num, 20, 64, 1)
print("features shape", features.shape)

### Save labels    
for i in tqdm(range(int(len(data))), unit='items'):
    for j in range(3):
        labels += (float(data[i][3]),)

labels = np.array(labels)

print("features:", features.shape)
print("labels:", labels.shape)

X_train, X_test, y_train, y_test = train_test_split(
		features, labels, test_size=0.10)
X_train, X_valid, y_train, y_valid = train_test_split(
		X_train, y_train, test_size=0.25)



# Print shapes of arrays that are imported
print('Data and modules loaded.')
print("train_features size:", X_train.shape)
print("train_labels size:", y_train.shape)
print("valid_features size:", X_valid.shape)
print("valid_labels size:", y_valid.shape)
print("test_features size:", X_test.shape)
print("test_labels size:", y_test.shape)

# the data, shuffled and split between train and test sets
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_valid /= 255
X_test  /= 255
X_train -= 0.5
X_valid -= 0.5
X_test  -= 0.5

# This is the shape of the image
input_shape = X_train.shape[1:]
print(input_shape, 'input shape')

# Set the parameters and print out the summary of the model
np.random.seed(1337)  # for reproducibility

batch_size = 64 # The lower the better
nb_classes = 1 # The output is a single digit: a steering angle
nb_epoch = 10 # The higher the better

# import model and wieghts if exists
try:
	with open('model.json', 'r') as jfile:
	    model = model_from_json(json.load(jfile))

	# Use adam and mean squared error for training
	model.compile("adam", "mse")

	# import weights
	model.load_weights('model.h5')

	print("Imported model and weights")

# If the model and weights do not exist, create a new model
except:
	# If model and weights do not exist in the local folder,
	# initiate a model

	# number of convolutional filters to use
	nb_filters1 = 16
	nb_filters2 = 8
	nb_filters3 = 4
	nb_filters4 = 2

	# size of pooling area for max pooling
	pool_size = (2, 2)

	# convolution kernel size
	kernel_size = (3, 3)

	# Initiating the model
	model = Sequential()

	# Starting with the convolutional layer
	# The first layer will turn 1 channel into 16 channels
	model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
	                        border_mode='valid',
	                        input_shape=input_shape))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 16 channels into 8 channels
	model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 8 channels into 4 channels
	model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 4 channels into 2 channels
	model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# Apply Max Pooling for each 2 x 2 pixels
	model.add(MaxPooling2D(pool_size=pool_size))
	# Apply dropout of 25%
	model.add(Dropout(0.25))

	# Flatten the matrix. The input has size of 360
	model.add(Flatten())
	# Input 360 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Input 16 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Input 16 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Apply dropout of 50%
	model.add(Dropout(0.5))
	# Input 16 Output 1
	model.add(Dense(nb_classes))

# Print out summary of the model
model.summary()

# Compile model using Adam optimizer 
# and loss computed by mean squared error
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

### Model training
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save the model.
# If the model.json file already exists in the local file,
# warn the user to make sure if user wants to overwrite the model.
if 'model.json' in os.listdir():
	print("The file already exists")
	print("Want to overwite? y or n")
	user_input = input()

	if user_input == "y":
		# Save model as json file
		json_string = model.to_json()

		with open('model.json', 'w') as outfile:
			json.dump(json_string, outfile)

			# save weights
			model.save_weights('./model.h5')
			print("Overwrite Successful")
	else:
		print("the model is not saved")
else:
	# Save model as json file
	json_string = model.to_json()

	with open('model.json', 'w') as outfile:
		json.dump(json_string, outfile)

		# save weights
		model.save_weights('./model.h5')
		print("Saved")
