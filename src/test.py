import lib.PrintImg
import lib.DataInput
import gc
import sys
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import scipy

import keras.optimizers as kerasopt
from keras.utils.np_utils import to_categorical

import pandas as pd
data_label = pd.read_csv("../data/stage1_labels.csv")

import re
data_label["graph_ID"]= data_label["Id"].map(lambda x: re.split("_",x)[0])
data_label["Zone"]= data_label["Id"].map(lambda x: re.split("_",x)[1][4:])

label = data_label.pivot(index='graph_ID', columns='Zone', values='Probability')
label.to_csv("../data/label_imputed.csv")
import numpy as np
label = label[[str(i) for i in range(1,18)]]
label_dict = dict()
for index, row in label.iterrows():
    label_dict[index] = row.tolist()

from os import walk, path
import sys

name_list = []
response = []
file_list = []
for (dirpath, dirnames, filenames) in walk("../data/"):
    sys.stderr.write("Directory path : " + dirpath + '\n')
    sys.stderr.write("Total number of files : " + str(len(filenames)) + '\n')
    file_index = 0
    for name in filenames:
        file_index += 1
        if (file_index % 100) == 1:
            sys.stderr.write("Current file : " + '[' + str(file_index) + '/' + str(len(filenames)) + ']' + '\n')
        if name[0] == '.':
            continue
        if re.split('\.', str(name))[-1] == 'aps':
            G_id = re.split('\.', str(name))[0]
            if label_dict.get(G_id) != None:
                name_list.append(G_id)
                file_list.append(path.join(dirpath, name))
                response.append(label_dict.get(G_id)[4])

# memory_limitation
file_list = file_list[0:800]
response = response[0:800]


train_file, test_file, Y_train, Y_test = train_test_split(file_list, response, test_size=0.2, random_state=4)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

angle = [0]
width = list(range(150,350))
height = list(range(300,500))
index = 0
sys.stderr.write("Road training data:" + '\n')
for file in train_file:
    if (index % 100) == 0:
        sys.stderr.write("Current file : " + '[' + str(index + 1) + '/' + str(len(train_file)) + ']' + '\n')
    graph = lib.DataInput.read_data(file)[width, :, :][:,height,:][:,:,angle]
    graph = graph/np.max(graph)
    if (index % 100) == 0:
        img = np.flipud(graph[:, :, 0].transpose())
        plt.imsave("../sample_train_chest/" + str(index) + '.png', img)

        X_train = np.zeros([len(train_file)] + list(graph.shape))
    X_train[index, :, :, :] = graph
    index += 1
sys.stderr.write("Training data done, shape : " + str(X_train.shape) + '\n')

index = 0
sys.stderr.write("Road test data:" + '\n')
for file in test_file:
    if (index % 100) == 0:
        sys.stderr.write("Current file : " + '[' + str(index + 1) + '/' + str(len(test_file)) + ']' + '\n')
    graph = lib.DataInput.read_data(file)[width, :, :][:,height,:][:,:,angle]
    graph = graph/np.max(graph)
    if index == 0:
        X_test = np.zeros([len(test_file)] + list(graph.shape))
    X_test[index, :, :, :] = graph
    index += 1
sys.stderr.write("test data done, shape : " + str(X_test.shape) + '\n')

from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten

#hyper parameters
num_classes = Y_train.shape[1]
num_train,height, width, depth = X_train.shape
batch_size = 32 # in each iteration, we consider 1 training examples at once
num_epochs = 20000 # we iterate 200 times over the entire training set
kernel_size = 6 # we will use 3x3 kernels throughout
pool_size = 4# we will use 2x2 pooling throughout
conv_depth_1 = 64 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 128 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.025 # dropout after pooling with probability 0.25
drop_prob_2 = 0.05 # dropout in the FC layer with probability 0.5
hidden_size = 128# the FC layer will have 512 neurons

inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
#conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
#drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(pool_1)
#conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)

conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(pool_2)
#conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)


#drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(pool_3)
hidden = Dense(hidden_size, activation='relu')(flat)
#drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(hidden)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

opt=kerasopt.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
#              optimizer='adam', # using the Adam optimiser
#              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
Y_test_pred = model.predict(X_test,verbose=1)  # Evaluate the trained model on the test set!

model.save("CNN_1")
np.savetxt("Y_test_pred.csv",Y_test_pred , delimiter=",")
np.savetxt("Y_test.csv",Y_test , delimiter=",")

accuracy = np.sum(abs(Y_test_pred-Y_test))/Y_test_pred.size

sys.stderr.write("Validation Accuracy : "+ str(accuracy))
