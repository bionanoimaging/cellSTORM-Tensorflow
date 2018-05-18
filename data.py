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

import h5py
from sklearn.model_selection import train_test_split

Examples = collections.namedtuple("Examples", "paths, inputs, targets, spikes, count, steps_per_epoch")

def norm_min_max(inputs):
     # min/max projection      - assume 0 as the lowest intensity
    min_inputs = np.min(inputs, axis=(1,2,3))
    proj01 = inputs-min_inputs[:, np.newaxis, np.newaxis, np.newaxis]
    max_inputs = np.max(proj01, axis=(1,2,3))
    proj01 = proj01/max_inputs[:, np.newaxis, np.newaxis, np.newaxis]
    
    return proj01
    
    

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
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        image = image-tf.reduce_min(image)
        image = image/tf.reduce_max(image)
        return image #(image + 1) / 2

        
def save_as_tif(inputs_np, outputs_np, targets_np, experiment_name, network_name):   
    """ Save images from results. """
    
     # create filedir according to the filename
    myfile_dir = ('./myresults/' + experiment_name + '_' + network_name)
    if not os.path.exists(myfile_dir):
        os.makedirs(myfile_dir)

    out_path_inputs = os.path.join(myfile_dir, experiment_name+'_inputs.tif')
    out_path_outputs = os.path.join(myfile_dir, experiment_name+'_outputs.tif')
    out_path_targets = os.path.join(myfile_dir, experiment_name+'_targets.tif')
    
    
    tifffile.imsave(out_path_inputs, inputs_np, append=True, bigtiff=True)
    tifffile.imsave(out_path_outputs, outputs_np, append=True, bigtiff=True)
    tifffile.imsave(out_path_targets, targets_np, append=True, bigtiff=True)
  
# synchronize seed for image operations so that we do the same operations to both
# input and output images
def transform(image, scale_size):
    r = image
    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.BILINEAR)
    return r
    

def merge_examples(example1, example2):
    # merge two datasets
    
    # define common variables
    count = example1.count
    steps_per_epoch = example1.steps_per_epoch
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
        count=count,
        steps_per_epoch=steps_per_epoch,
    )

# load database as h5 file from disk
def load_examples_h5(filename, batch_size, is_normalize = False):
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
    
    
    # convert Numpy to Tensorflow's tensor
    if(0):
        # this is not possible! toooo much memory! 
        patches_tensor = tf.transpose(tf.convert_to_tensor(patches), perm=[1, 2, 3, 0])
        heatmaps_tensor = tf.transpose(tf.convert_to_tensor(heatmaps), perm=[1, 2, 3, 0])
        spikes_tensor = tf.transpose(tf.convert_to_tensor(spikes), perm=[1, 2, 3, 0])
    
    
    steps_per_epoch = int(math.ceil(count / batch_size))
    
    print('Reading finished.')
    
    print('Number of Training Examples: %d' % count)
    
    return Examples(
        paths=filename,
        inputs=patches,
        targets=heatmaps,
        spikes=spikes,
        count=count,
        steps_per_epoch=steps_per_epoch,
    )

# load image-pairs from disk 
def load_examples(input_dir, scale_size, batch_size, mode):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
    else:
        print("Found " + str(len(input_paths)) + " images...")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 1])

      
        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        inputs = preprocess(raw_input[:,0:(width/3),:])
        spikes  = preprocess(raw_input[:,(width/3):(2*width/3),:])
        targets = preprocess(raw_input[:,(2*width/3):-1:,:])
 



    with tf.name_scope("input_images"):
        input_images = transform(inputs, scale_size, )

    with tf.name_scope("target_images"):
        target_images = transform(targets, scale_size)

    with tf.name_scope("spikes_images"):
        spikes_images = transform(spikes, scale_size)


    
    paths_batch, inputs_batch, targets_batch, spikes_batch = tf.train.batch([paths, input_images, target_images, spikes_images], batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        spikes=spikes_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )



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
        
    def loadDummy(self):
     
        # assign dummy variables according to "load_examples"
        inputs_batch = tf.zeros(shape = (1, self.scale_size, self.scale_size, 1))
        targets_batch = inputs_batch 
        count_batch =  self.__len__()
        steps_per_epoch = 1
        paths_batch = self.dir_AB 
        
        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            spikes=targets_batch*0,
            count=count_batch,
            steps_per_epoch=steps_per_epoch,
        )

    def __getitem__(self, index):
        # read frame
        frame = self.videogen.next()

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
        
        
        # Pre-Process: Normalize and zero-center
        frame_mean = frame_mean-np.min(frame_mean)
        frame_mean  = frame_mean/np.max(frame_mean)
        frame_mean = frame_mean * 2. - 1.
        
        start_x = np.int32(self.xcenter-self.roisize/2)
        end_x = np.int32(self.xcenter+self.roisize/2)
        start_y = np.int32(self.ycenter-self.roisize/2)
        end_y = np.int32(self.ycenter+self.roisize/2)
        
        frame_crop = frame_mean[start_x:end_x, start_y:end_y]
        #npad = ((self.padwidth, self.padwidth), (self.padwidth, self.padwidth), (0, 0))
        #frame_pad = np.pad(frame_crop, npad, 'reflect')

        # resize to scale_size
        frame_crop = scipy.misc.imresize(frame_crop, size = (self.scale_size, self.scale_size), interp='bilinear', mode='F')
        frame_crop = np.expand_dims(np.expand_dims(frame_crop, axis = 0), axis = 3) # add zero-batch dimension and color-channel
        
        return frame_crop
    
    
    def __len__(self):
        videometadata = skvideo.io.ffprobe(self.dir_AB)
        #print(videometadata)
        #print(self.dir_AB)
        num_frames = np.int(videometadata['video']['@nb_frames'])

        return num_frames

    def name(self):
        return 'VideoDataset'
