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
import matplotlib.pyplot as plt

def normalize_max_std_mean(inputs):
    # min/max projection      - assume 0 as the lowest intensity
    min_inputs = np.min(inputs, axis=(1,2,3))
    proj01 = inputs-min_inputs[:, np.newaxis, np.newaxis, np.newaxis]
    max_inputs = np.max(proj01, axis=(1,2,3))
    proj01 = proj01/max_inputs[:, np.newaxis, np.newaxis, np.newaxis]
    
    
    
    mean_inputs = np.mean(proj01, axis=(1,2,3))
    std_inputs = np.std(proj01, axis=(1,2,3))
    
    print("Max: ", str(np.max(max_inputs)), "Mean: ", str(mean_inputs), ", Std: ", str(std_inputs))
    
    inputs_norm = (proj01-mean_inputs)/std_inputs
    
    return inputs_norm
        


if(1):
    filename = './cellstorm_data.h5'
    is_normalize = True
    #BATCH_SIZE = 4
    # Load training data and divide it to training and validation sets
    # borrowed from Deep-STORM
    matfile = h5py.File(filename, 'r')
    
    # get the matrices
    patches = np.float32(np.array(matfile['patches']))
    heatmaps = np.float32(np.array(matfile['heatmaps']))
    spikes = np.float32(np.array(matfile['spikes']))
    
    # arrange to TF coordinate system
    patches = np.expand_dims(patches, axis = 3)
    heatmaps = np.expand_dims(heatmaps, axis = 3)
    spikes = np.expand_dims(spikes, axis = 3)
    
    if(is_normalize):
         #===================== Training set normalization ==========================
        # normalize training images to be in the range [0,1] and calculate the 
        # training set mean and std

        # resulting normalized training images
        patches = normalize_max_std_mean(patches)
        heatmaps = normalize_max_std_mean(heatmaps)
        spikes = normalize_max_std_mean(spikes)
        
    else:            
           # preprocess data 255->1->0..1 -> -1..1 #TODO: Alternativelly: Whitening?!
        patches = (2*patches/255.)-1
        heatmaps = (2*heatmaps/255.)-1
        spikes = (2*spikes/255.)-1
    
    count = patches.shape[0]
    # randomize the order of the data
    # assuming data in order: [N_smaples, Width, Height, Color-channels]
    shuffle_order = np.arange(count)
    shuffle_order = np.random.shuffle(shuffle_order)
    
    patches = patches[shuffle_order, :, :]
    heatmaps = heatmaps[shuffle_order, :, :]
    spikes = spikes[shuffle_order, :, :]

    print('Reading finished.')
    
    print('Number of Training Examples: %d' % X.shape[1])
    
if(0):
    input_dir = '/home/diederich/Documents/STORM/DATASET_NN/04_UNPROCESSED_RAW_HW/MOV_2018_02_16_09_09_49_ISO3200_texp_1_85_lines_combined/test'
    #input_dir = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/STORM/MATLAB/cellSTORM-KERAS/images/2017-12-18_18.29.45.mp4_256_Tirf_v2_from_video_v2_fakeB_2k.tif_bilinear_smallpatch'
    scale_size = 256
    batch_size = 4
    mode = 'train'
    reload(data)
    examples = data.load_examples(input_dir, scale_size, batch_size, mode)
     
    
