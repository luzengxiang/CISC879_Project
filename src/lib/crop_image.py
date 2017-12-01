from os import walk,path
import sys
import lib.DataInput
import re
import pandas as pd
from sklearn.cross_validation import train_test_split
from lib.func import convert_to_grayscale as convert_to_grayscale
from matplotlib import pyplot as plt
import os
import numpy as np

from keras.utils.np_utils import to_categorical
def crop(img, crop_list):
    x_coord = crop_list[0]
    y_coord = 660-crop_list[1]
    width = crop_list[2]
    height = crop_list[3]
    cropped_img = img[x_coord:x_coord + width, y_coord - height:y_coord]

    return cropped_img


def import_eps(data_folder,label_file,zone,savepath = '../sample_train/',sep = False):
    savepath = os.path.splitext(savepath)[0] + '/'
    if os.path.exists(savepath) == False:
        os.mkdir(savepath)

    data_label = pd.read_csv(label_file)
    data_label["graph_ID"] = data_label["Id"].map(lambda x: re.split("_", x)[0])
    data_label["Zone"] = data_label["Id"].map(lambda x: re.split("_", x)[1][4:])
    label = data_label.pivot(index='graph_ID', columns='Zone', values='Probability')
    label = label[[str(i) for i in range(1, 18)]]
    label_dict = dict()
    for index, row in label.iterrows():
        label_dict[index] = row.tolist()


    name_list=[]
    response = []
    file_list = []

    sector_crop_list = [[ 50, 50, 250, 250], # sector 1
                    [  0,   0, 250, 250], # sector 2
                    [250, 50, 250, 250], # sector 3
                    [250,   0, 250, 250], # sector 4
                    [150, 150, 250, 250], # sector 5/17
                    [200, 100, 250, 250], # sector 6
                    [200, 150, 250, 250], # sector 7
                    [250,  50, 250, 250], # sector 8
                    [250, 150, 250, 250], # sector 9
                    [300, 200, 250, 250], # sector 10
                    [400, 100, 250, 250], # sector 11
                    [350, 200, 250, 250], # sector 12
                    [410,   0, 250, 250], # sector 13
                    [410, 200, 250, 250], # sector 14
                    [410,   0, 250, 250], # sector 15
                    [410, 200, 250, 250], # sector 16
                   ]


    zone_crop_list = [[  # threat zone 1
    sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], None,
    None, None, sector_crop_list[2], sector_crop_list[2],
    sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], None,
    None, sector_crop_list[0], sector_crop_list[0],
    sector_crop_list[0]],

    [  # threat zone 2
        sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], None,
        None, None, sector_crop_list[3], sector_crop_list[3],
        sector_crop_list[3], sector_crop_list[3], sector_crop_list[3],
        None, None, sector_crop_list[1], sector_crop_list[1],
        sector_crop_list[1]],

    [  # threat zone 3
        sector_crop_list[2], sector_crop_list[2], sector_crop_list[2],
        sector_crop_list[2], None, None, sector_crop_list[0],
        sector_crop_list[0], sector_crop_list[0], sector_crop_list[0],
        sector_crop_list[0], sector_crop_list[0], None, None,
        sector_crop_list[2], sector_crop_list[2]],

    [  # threat zone 4
        sector_crop_list[3], sector_crop_list[3], sector_crop_list[3],
        sector_crop_list[3], None, None, sector_crop_list[1],
        sector_crop_list[1], sector_crop_list[1], sector_crop_list[1],
        sector_crop_list[1], sector_crop_list[1], None, None,
        sector_crop_list[3], sector_crop_list[3]],

    [  # threat zone 5
        sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
        sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
        sector_crop_list[4], sector_crop_list[4],
        None, None, None, None, None, None, None, None],

    [  # threat zone 6
        sector_crop_list[5], None, None, None, None, None, None, None,
        sector_crop_list[6], sector_crop_list[6], sector_crop_list[5],
        sector_crop_list[5], sector_crop_list[5], sector_crop_list[5],
        sector_crop_list[5], sector_crop_list[5]],

    [  # threat zone 7
        sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
        sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
        sector_crop_list[6], sector_crop_list[6],
        None, None, None, None, None, None, None, None],

    [  # threat zone 8
        sector_crop_list[7], sector_crop_list[7], None, None, None,
        None, None, sector_crop_list[9], sector_crop_list[9],
        sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
        sector_crop_list[7], sector_crop_list[7], sector_crop_list[7],
        sector_crop_list[7]],

    [  # threat zone 9
        sector_crop_list[8], sector_crop_list[8], sector_crop_list[7],
        sector_crop_list[7], sector_crop_list[7], None, None, None,
        sector_crop_list[8], sector_crop_list[8], None, None, None,
        None, sector_crop_list[9], sector_crop_list[8]],

    [  # threat zone 10
        sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
        sector_crop_list[9], sector_crop_list[9], sector_crop_list[7],
        sector_crop_list[9], None, None, None, None, None, None, None,
        None, sector_crop_list[9]],

    [  # threat zone 11
        sector_crop_list[10], sector_crop_list[10], sector_crop_list[10],
        sector_crop_list[10], None, None, sector_crop_list[11],
        sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
        sector_crop_list[11], None, sector_crop_list[10],
        sector_crop_list[10], sector_crop_list[10], sector_crop_list[10]],

    [  # threat zone 12
        sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
        sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
        sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
        sector_crop_list[11], sector_crop_list[11], None, None,
        sector_crop_list[11], sector_crop_list[11], sector_crop_list[11]],

    [  # threat zone 13
        sector_crop_list[12], sector_crop_list[12], sector_crop_list[12],
        sector_crop_list[12], None, None, sector_crop_list[13],
        sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
        sector_crop_list[13], None, sector_crop_list[12],
        sector_crop_list[12], sector_crop_list[12], sector_crop_list[12]],

    [  # sector 14
        sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
        sector_crop_list[13], sector_crop_list[13], None,
        sector_crop_list[13], sector_crop_list[13], sector_crop_list[12],
        sector_crop_list[12], sector_crop_list[12], None, None, None,
        None, None],

    [  # threat zone 15
        sector_crop_list[14], sector_crop_list[14], sector_crop_list[14],
        sector_crop_list[14], None, None, sector_crop_list[15],
        sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
        None, sector_crop_list[14], sector_crop_list[14], None,
        sector_crop_list[14], sector_crop_list[14]],

    [  # threat zone 16
        sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
        sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
        sector_crop_list[14], sector_crop_list[14], sector_crop_list[14],
        sector_crop_list[14], sector_crop_list[14], None, None, None,
        sector_crop_list[15], sector_crop_list[15]],

    [  # threat zone 17
        None, None, None, None, None, None, None, None,
        sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
        sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
        sector_crop_list[4], sector_crop_list[4]]]

    for (dirpath, dirnames, filenames) in walk(data_folder):
        sys.stderr.write("Scan files:" + '\n')
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
                    response.append(label_dict.get(G_id)[zone-1])
    if sep:
        train_file, test_file, Y_train, Y_test = train_test_split(file_list, response, test_size=0.2, random_state=4)


        sys.stderr.write("Road training data:" + '\n')
        index = 0
        for file in train_file:
            if (index % 100) == 0:
                sys.stderr.write("Current file : " + '[' + str(index + 1) + '/' + str(len(train_file)) + ']' + '\n')
            graph =  convert_to_grayscale(lib.DataInput.read_data(file))

            zone_crop_list_i = zone_crop_list[zone-1]
            graph_number = sum([i is not None for i in zone_crop_list_i])


            if index == 0:
                X_train = np.zeros([len(train_file),250,250,graph_number])
            graph_out_number = 0
            for i,crop_zone in enumerate(zone_crop_list_i):
                if crop_zone is not None:
                    X_train[index, :, :,graph_out_number] = crop(graph[:,:,i],crop_zone)
                    graph_out_number += 1

            #save sample image from training set
            if (index % 100) == 0:
                for g_i in range(0,graph_number):
                    img = np.flipud(X_train[index,:,:, g_i].transpose())
                    plt.imsave(savepath + "zone"+"_"+str(zone)+'_'+str(index)+"_"+str(g_i)+ '.png', img)
            index += 1

        sys.stderr.write("Training data done, shape : " + str(X_train.shape) + '\n')

        index = 0
        for file in test_file:
            if (index % 100) == 0:
                sys.stderr.write("Current file : " + '[' + str(index + 1) + '/' + str(len(test_file)) + ']' + '\n')
            graph = convert_to_grayscale(lib.DataInput.read_data(file))

            zone_crop_list_i = zone_crop_list[zone - 1]
            graph_number = sum([i is not None for i in zone_crop_list_i])

            if index == 0:
                X_test = np.zeros([len(test_file), 250, 250,graph_number])
            graph_out_number = 0
            for i,crop_zone in enumerate(zone_crop_list_i):
                if crop_zone is not None:
                    X_test[index,  :, :,graph_out_number] = crop(graph[:,:,i], crop_zone)
                    graph_out_number += 1
            index += 1
        sys.stderr.write("Training data done, shape : " + str(X_train.shape) + '\n')
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)
        return X_train,X_test,Y_train,Y_test
    else:
        sys.stderr.write("Road  data:" + '\n')
        index = 0
        for file in file_list:
            if (index % 100) == 0:
                sys.stderr.write("Current file : " + '[' + str(index + 1) + '/' + str(len(file_list)) + ']' + '\n')
            graph = convert_to_grayscale(lib.DataInput.read_data(file))

            zone_crop_list_i = zone_crop_list[zone - 1]
            graph_number = sum([i is not None for i in zone_crop_list_i])

            if index == 0:
                X = np.zeros([len(file_list), 250, 250, graph_number])
            graph_out_number = 0
            for i, crop_zone in enumerate(zone_crop_list_i):
                if crop_zone is not None:
                    X[index, :, :, graph_out_number] = crop(graph[:, :, i], crop_zone)
                    graph_out_number += 1

            # save sample image from training set
            if (index % 100) == 0:
                for g_i in range(0, graph_number):
                    img = np.flipud(X[index, :, :, g_i].transpose())
                    plt.imsave(savepath + "zone" + "_" + str(zone) + '_' + str(index) + "_" + str(g_i) + '.png', img)
            index += 1

        sys.stderr.write("Data done, shape : " + str(X.shape) + '\n')
        Y = to_categorical(response)
        return X,Y