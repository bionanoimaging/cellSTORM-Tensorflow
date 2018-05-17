#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:33:10 2018

@author: bene
"""

import tensorflow as tf
import h5py
import numpy as np
import data as data

if(1):
    filename = './cellstorm_data.h5'
    BATCH_SIZE = 4
    # Load training data and divide it to training and validation sets
    # borrowed from Deep-STORM
    matfile = h5py.File(filename, 'r')
    matfile = h5py.File(filename, 'r')
    
    X_train = np.float32(np.array(matfile['patches']))
    Y_train = np.float32(np.array(matfile['heatmaps']))
    
    # randomize the order of the data
    # assuming data in order: [N_smaples, Width, Height, Color-channels]
    shuffle_order = np.arange(X_train.shape[0])
    shuffle_order = np.random.shuffle(shuffle_order)
    
    X_train = X_train[shuffle_order, :, :]
    Y_train = Y_train[shuffle_order, :, :]
    
    # convert Numpy to Tensorflow's tensor
    X_train_tensor = tf.transpose(tf.convert_to_tensor(X_train), perm=[1, 2, 3, 0])
    Y_train_tensor = tf.transpose(tf.convert_to_tensor(Y_train), perm=[1, 2, 3, 0])
    
    # create input pipeline
    X_train_tensor = tf.train.slice_input_producer([X_train_tensor], shuffle=False)
    Y_train_tensor = tf.train.slice_input_producer([Y_train_tensor], shuffle=False)
    
    
    # pack into shuffle batch
    batch = tf.train.shuffle_batch([X_train_tensor, Y_train_tensor],
                                   batch_size=BATCH_SIZE, capacity=BATCH_SIZE * 5,
                                   min_after_dequeue=BATCH_SIZE * 3)
    
    
    print('Reading finished.')
    
    print('Number of Training Examples: %d' % X_train.shape[1])
    
if(0):
    input_dir = '/home/diederich/Documents/STORM/DATASET_NN/04_UNPROCESSED_RAW_HW/MOV_2018_02_16_09_09_49_ISO3200_texp_1_85_lines_combined/test'
    #input_dir = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/STORM/MATLAB/cellSTORM-KERAS/images/2017-12-18_18.29.45.mp4_256_Tirf_v2_from_video_v2_fakeB_2k.tif_bilinear_smallpatch'
    scale_size = 256
    batch_size = 4
    mode = 'train'
    reload(data)
    examples = data.load_examples(input_dir, scale_size, batch_size, mode)
     
    
