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

def preprocess(image):
    with tf.name_scope("preprocess"):
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
    
    
def save_images(fetches, output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

# load database as h5 file from disk
def load_examples_h5(filename, scale_size, batch_size, mode):
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
   
    return batch
    

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
        frame_mean = (frame_mean/255.0)
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
