#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:38:25 2018

@author: useradmin
"""
import tensorflow as tf
import numpy as np
import tifffile
import os
import glob
import math 
import collections
import os.path

import skvideo.io
import scipy.misc
import scipy.ndimage
import numpy as np
import cv2

import h5py
from sklearn.model_selection import train_test_split

Examples = collections.namedtuple("Examples", "paths, inputs, targets, spikes, count")

# want to normalize the data by its mean and std? not a good adia probalby..
is_normalize = False

def norm_min_max(inputs):
    if len(inputs.shape)==3:
         min_inputs = np.min(inputs)
         proj01 = inputs-min_inputs
         max_inputs = np.max(proj01)
         proj01 = proj01/max_inputs
    else:             
         min_inputs = np.min(inputs, axis=(1,2,3))
         proj01 = inputs-min_inputs[:, np.newaxis, np.newaxis, np.newaxis]
         max_inputs = np.max(proj01, axis=(1,2,3))
         proj01 = proj01/max_inputs[:, np.newaxis, np.newaxis, np.newaxis]
    
    return proj01
    
def norm_min_max_tf(image):
    image = image-tf.reduce_min(image)
    image = image/tf.reduce_max(image)
    return image

def normalize_std_mean(inputs):
   
    # get the mean and std from the normalized inputs to do whitening
    mean_inputs = np.mean(inputs, axis=(1,2,3))
    std_inputs = np.std(inputs, axis=(1,2,3))
    
    print("Mean: ", str(mean_inputs), ", Std: ", str(std_inputs))
    
    inputs_norm = (inputs-mean_inputs[:, np.newaxis, np.newaxis, np.newaxis])/std_inputs[:, np.newaxis, np.newaxis, np.newaxis]
    
    return inputs_norm
        

def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
    # [-1, 1] => [0, 1]
    image = image-np.min(image)
    image = image/np.max(image)
    return image #(image + 1) / 2

def deprocess_tf(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        if(is_normalize):
            image = image-tf.reduce_min(image)
            image = image/tf.reduce_max(image)
        else:
            image = (image + 1) / 2
        return image

        
def save_as_tif(inputs_np, outputs_np, experiment_name, network_name):   
    """ Save images from results. """
    
     # create filedir according to the filename
    myfile_dir = ('./myresults/' + experiment_name + '_' + network_name)
    if not os.path.exists(myfile_dir):
        os.makedirs(myfile_dir)

    out_path_inputs = os.path.join(myfile_dir, experiment_name+'_inputs.tif')
    out_path_outputs = os.path.join(myfile_dir, experiment_name+'_outputs.tif')
        
    tifffile.imsave(out_path_inputs, np.uint16(inputs_np), append=True, bigtiff=True) #compression='lzw', 
    tifffile.imsave(out_path_outputs, np.uint16(outputs_np*3),  append=True, bigtiff=True) # int saves space
    
def merge_examples(example1, example2):
    # merge two datasets
    
    # define common variables
    count = example1.count
    paths = ''
    
    # compbine inputs
    inputs = np.concatenate((example1.inputs, example2.inputs), axis=0)
    # compbine spikes
    spikes = np.concatenate((example1.spikes, example2.spikes), axis=0)
    # compbine targets
    targets = np.concatenate((example1.targets, example2.targets), axis=0)
    
    return Examples(
        paths=paths,
        inputs=inputs,
        targets=targets,
        spikes=spikes,
        count=count
    )
    
def limit_numsamples(examples, num_samples):
    # merge two datasets
        
    # limits inputs
    inputs = examples.inputs[0:num_samples,:,:,:]
    # compbine spikes
    spikes = examples.spikes[0:num_samples,:,:,:]
    # compbine targets
    targets = examples.targets[0:num_samples,:,:,:]
    
    return Examples(
        paths='',
        inputs=inputs,
        targets=targets,
        spikes=spikes,
        count=num_samples,
    )    

# load database as h5 file from disk
def load_examples_h5(filename, batch_size, is_normalize = is_normalize):
    #filename = './cellstorm_data.h5'
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
 
    # bring data to 0..1
    patches = norm_min_max(patches)
    heatmaps = norm_min_max(heatmaps)
    spikes = norm_min_max(spikes)
    
    
    if(is_normalize):
         #===================== Training set normalization ==========================
        # normalize training images to be in the range [0,1] and calculate the 
        # training set mean and std

        # resulting normalized training images
        patches = normalize_std_mean(patches)
        heatmaps = normalize_std_mean(heatmaps)
        spikes = normalize_std_mean(spikes)
        
    else:            
           # preprocess data 255->1->0..1 -> -1..1 #TODO: Alternativelly: Whitening?!
        patches = preprocess(patches)
        heatmaps = preprocess(heatmaps)
        spikes = preprocess(spikes)
   
    
    count = patches.shape[0]
    
    # do shuffeling here doesn't make sense if you load mulitple datasets, therefore do shuffleing at training time!
    if(0):
        
        # randomize the order of the data
        # assuming data in order: [N_smaples, Width, Height, Color-channels]
        shuffle_order = np.arange(count)
        shuffle_order = np.random.shuffle(shuffle_order)
        
        patches = patches[shuffle_order, :, :, :]
        heatmaps = heatmaps[shuffle_order, :, :, :]
        spikes = spikes[shuffle_order, :, :, :]
        
        # weird that it adds an additional axis..
        patches = np.squeeze(patches, axis=0)
        heatmaps = np.squeeze(heatmaps, axis=0)
        spikes = np.squeeze(spikes, axis=0)
    

    print('Reading finished.')
    
    print('Number of Training Examples: %d' % count)
    
    return Examples(
        paths=filename,
        inputs=patches,
        targets=heatmaps,
        spikes=spikes,
        count=count
    )


def write_to_csv(data, filename):
    import csv
  
    header = ['id', 'frame', 'x [nm]','y [nm]', 'sigma [nm]','intensity [photon]','offset [photon]','bkgstd [photon]','chi2','uncertainty [nm]']
    
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
         
        data_id = np.int32(data[0])
        data_frame = np.int32(data[1])
        data_y = np.float32(data[2])
        data_x = np.float32(data[3])
        data_intensity = np.float32(data[4])
        
        writer.writerow(header)    
        for i in range(data_id.shape[0]):
            data_row = list((data_id[i], data_frame[i], data_x[i], data_y[i], 0, data_intensity[i], 0))
            #print(data_row)
            writer.writerow(data_row)    
    
    csvfile.close()

## create class for frame-to-frame reading
class VideoReader:
    
    def __init__(self, dataroot, scale_size, roisize, xcenter, ycenter):
        self.dir_AB = dataroot
        
        # open videoreader
        self.videogen = skvideo.io.vreader(self.dir_AB )
        
        # define roisize and center where each frame will be extracted
        self.roisize = roisize #512
        self.padwidth = 0# padwidth
        self.xcenter = xcenter
        self.ycenter =  ycenter
        self.scale_size = scale_size
        

     
        # assign dummy variables according to "load_examples"
        self.firstframe = next(self.videogen)#next()
        self.count =  self.__len__()
        
        self.refPt = []
                        
        
        
     
    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
         
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            print(self.refPt)
    	        
    
    def select_ROI(self):
        # load the image, clone it, and setup the mouse callback function
        image = cv2.cvtColor( self.firstframe*2, cv2.COLOR_RGB2GRAY )
        image = cv2.equalizeHist(image)
        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)

            
        # keep looping until the 'q' key is pressed
        while True:
            
            # display the image and wait for a keypress  
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            # if there are two reference points, then crop the region of interest
            # from teh image and display it
            if len(self.refPt) == 1:
                #roi = clone[self.refPt[0][0]-self.roisize/2:self.refPt[0][0]+self.roisize/2, self.refPt[0][1]-self.roisize/2:self.refPt[0][1]+self.roisize/2]
                break

        # close all open windows
        cv2.destroyAllWindows()
        
        self.xcenter = self.refPt[0][1]
        self.ycenter =  self.refPt[0][0]        

    def __getitem__(self, index):
        # read frame
        frame = next(self.videogen)#self.videogen.next()

        # if no center is chosen, select the videos center
        if self.xcenter == -1:
            self.xcenter = int(frame.shape[0]/2)
            print('New xcenter: ' + str(self.xcenter))
        if self.ycenter == -1:
            self.ycenter = int(frame.shape[1]/2)
            print('New ycenter: ' + str(self.ycenter))
        if self.roisize == -1:
            self.roisize = int(np.min(frame.shape[0:1]))
            print('New roisize: ' + str(self.roisize))
        
        

        # crop frame to ROI
        frame_mean = np.mean(frame, axis=2)
       
        # Calculate the coordinates for the ROI
        start_x = np.int32(self.xcenter-self.roisize/2)
        end_x = np.int32(self.xcenter+self.roisize/2)
        start_y = np.int32(self.ycenter-self.roisize/2)
        end_y = np.int32(self.ycenter+self.roisize/2)
        
        # crop ROI
        if(frame_mean.shape[0]<=self.roisize):
            frame_crop = frame_mean
        else:
            frame_crop = frame_mean[start_x:end_x, start_y:end_y]

        #npad = ((self.padwidth, self.padwidth), (self.padwidth, self.padwidth), (0, 0))
        #frame_pad = np.pad(frame_crop, npad, 'reflect')


        # Pre-Process: Normalize and zero-center
        #frame_crop = frame_crop-np.min(frame_crop)
        try:
            frame_crop_processed  = frame_crop/np.max(frame_crop)
        except:
            print('Frame has shape: ', frame_crop.shape)
                
        if(1):
            frame_crop_processed = preprocess(frame_crop_processed)
        else:
            frame_crop_processed = (frame_crop_processed-np.mean(frame_crop_processed))/np.std(frame_crop_processed)
        
        # resize to scale_size
        frame_crop_processed = scipy.misc.imresize(frame_crop_processed, size = (self.scale_size, self.scale_size), interp='bilinear', mode='F')
        frame_crop_processed = np.expand_dims(np.expand_dims(frame_crop_processed, axis = 0), axis = 3) # add zero-batch dimension and color-channel
        
        
        return frame_crop, frame_crop_processed
    
    
    def __len__(self):
        videometadata = skvideo.io.ffprobe(self.dir_AB)
        #print(videometadata)
        #print(self.dir_AB)
        num_frames = np.int(videometadata['video']['@nb_frames'])

        return num_frames

    def name(self):
        return 'VideoDataset'
