# -*- coding: utf-8 -*-

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:24:20 2018

@author: useradmin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import matplotlib.pyplot as plt

Model = collections.namedtuple("Model", "outputs, outputs_psf, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_sparse_L1, gen_grads_and_vars, targets, inputs, train")

psf_size = 31
psf_sigma = 4 # corresponds to 80nm effective pixelsize with 5x magnification of the video => FWHM ~ 15 pixel

is_spikes = True
#  Define a matlab like gaussian 2D filter
def matlab_style_gauss2D(shape=(7,7),sigma=1):
    """ 
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma]) 
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.float32(np.exp( -(x*x + y*y) / (2.*sigma*sigma) ))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = np.float32(h*2.0)
    return h

# Expand the filter dimensions
psf_heatmap = matlab_style_gauss2D(shape = (psf_size,psf_size),sigma=psf_sigma)
gfilter = tf.reshape(psf_heatmap, [psf_size, psf_size, 1, 1])
#plt.imshow(np.squeeze(psf_heatmap)), plt.show()



def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels, stride=2):
    """ Convolution. """
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        conv_filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT",  constant_values=0)
        conv = tf.nn.conv2d(padded_input, conv_filter, [1, stride, stride, 1], padding="VALID")
        return conv


def gen_deconv(batch_input, out_channels):
     ''' Transposed Convolution. '''
     with tf.variable_scope("deconv"):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if True:
            # this is very likely responssible for the checkerboard
            # remove checkerboard artifact have a look at the distill paper" #

            _b, h, w, _c = batch_input.shape
            resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            print(resized_input)
            return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def gen_deconv_npu_mod(batch_input, out_channels):
     ''' Transposed Convolution. '''
     with tf.variable_scope("deconv"):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        
        _b, h, w, _c = batch_input.shape
        _c_mod = 16 #int(int(_c)/4) # also take into account that stride only takes mod 16 strides! 
        w_mod = 8 #int(int(_c)/4) # also take into account that stride only takes mod 16 strides! 
        h_mod = 8 #int(int(_c)/4) # also take into account that stride only takes mod 16 strides! 
        # image_resize conflicts with the dimensions, so trying to split and later concat the layers
        if(int(_c) <= 256):

            for i in range(0,int(int(_c)/_c_mod)):
                batch_input_sub = tf.slice(batch_input, [0,0,0,i*_c_mod], [_b, h, w, _c_mod])



                if(int(w) > _c_mod):
                    # now it's getting really weird, trying to further reduce the size of the images by patches 
                    for w_i in range(0,int(int(w)/w_mod)):
                        batch_input_sub_patch_w = tf.slice(batch_input_sub, [0,0,w_i*w_mod,0], [_b, h, w_mod, _c_mod])

                        for h_i in range(0,int(int(h)/h_mod)):
                            batch_input_sub_patch_h = tf.slice(batch_input_sub_patch_w, [0,h_i*h_mod,0,0], [_b, h_mod, w_mod, _c_mod])
                            resized_input_sub_patch_h = tf.image.resize_images(batch_input_sub_patch_h, [h_mod * 2, w_mod * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                            print(resized_input_sub_patch_h)
                            
                            if(h_i==0):
                                resized_input_sub_h = resized_input_sub_patch_h
                                
                            else:
                                resized_input_sub_h  = tf.concat([resized_input_sub_h, resized_input_sub_patch_h], axis=1)

                            

                        if(w_i==0):
                            resized_input_sub = resized_input_sub_h
                        else:
                            resized_input_sub = tf.concat([resized_input_sub, resized_input_sub_h], axis=2)
    
                else:
                    resized_input_sub = tf.image.resize_images(batch_input_sub, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                
                

                if(i==0):
                    resized_input = resized_input_sub
                else:
                    resized_input = tf.concat([resized_input, resized_input_sub], axis=-1)
                
        else:
            resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)            
            

        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        

def gen_deconv_npu_mod_2(batch_input, out_channels):
     ''' Transposed Convolution. '''
     with tf.variable_scope("deconv"):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        
        _b, h, w, _c = batch_input.shape

        # image_resize conflicts with the dimensions, so trying to split and later concat the layers
        if(int(_c) <= 256):
            
            _c_mod = 256-_c
            print(_c_mod)
            print("Size was: "+str(batch_input)) 
            pseudo_patch = tf.zeros( [_b, h, w, _c_mod])
            batch_input = tf.concat([batch_input, pseudo_patch], axis=-1)
            print("Size is: "+str(batch_input))
            batch_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            print("Size is now: "+str(batch_input))
            resized_input = tf.slice(batch_input, [0,0,0,0], [_b, h, w, _c])
                
        else:
            resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
        print(resized_input)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)


# create U-NET generator as kind of a auto-encoder to filter the images
def create_generator(generator_inputs, generator_outputs_channels, NGF, is_training=True):
    layers = []
    

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, NGF)
        layers.append(output)

    layer_specs = [
        NGF * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        NGF * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        NGF * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        NGF * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        NGF * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        NGF * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        NGF * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    if is_training:
        dropout_prob = 0.5
    else:
        dropout_prob = 0. # test it for the NPU from HUAWEI (no dropout in their api yet)
            
    layer_specs = [
        (NGF * 8, dropout_prob),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (NGF * 8, dropout_prob),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (NGF * 8, dropout_prob),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (NGF * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (NGF * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (NGF * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (NGF, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            print(out_channels)

            # try to circumvent the dimension problem on HiAi
            output = gen_deconv_npu_mod(rectified, out_channels)
                
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]



def create_discriminator(discrim_inputs, discrim_targets, NDF):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, NDF, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = NDF * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]




def create_model(inputs, targets_raw, NDF, NGF, EPS, GAN_weight, L1_weight, L1_sparse_weight, Adam_LR, Adam_beta1, is_training=True):
    
    with tf.variable_scope("generator"):
        out_channels = int(targets_raw.get_shape()[-1])
        outputs = create_generator(inputs, out_channels, NGF, is_training)
        
        # generate the heatmap corresponding to the predicted spikes
        
        
        
    if(True):
        # Convolution with padding to get PSF where spikes would lie. 
        padsize = np.int8(psf_size / 2)
        outputs_padded = tf.pad(outputs, [[0, 0], [padsize, padsize], [padsize, padsize], [0, 0]], mode="REFLECT")
        outputs_psf = tf.nn.conv2d(outputs_padded, gfilter, [1, 1, 1, 1], padding="VALID")
    else:
        outputs_psf = tf.nn.conv2d(outputs, gfilter, strides=(1, 1, 1, 1), padding="SAME")

    # do the same trick for the targets - test using same PSF 
    if(is_spikes):
        padsize = np.int8(psf_size / 2)
        targets_padded = tf.pad(targets_raw, [[0, 0], [padsize, padsize], [padsize, padsize], [0, 0]], mode="REFLECT")
        targets = tf.nn.conv2d(targets_padded, gfilter, [1, 1, 1, 1], padding="VALID")
    else:
        targets = targets_raw
    

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets, NDF)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs_psf, NDF)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
#        discrim_loss = tf.reduce_mean(predict_real + predict_fake)

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        #gen_loss_GAN = tf.reduce_mean(predict_fake)
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs_psf))
        gen_loss_sparse_L1 = tf.reduce_mean(tf.abs(outputs))
        gen_loss = gen_loss_GAN * GAN_weight + gen_loss_L1 * L1_weight + gen_loss_sparse_L1 * L1_sparse_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(Adam_LR, Adam_beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(Adam_LR, Adam_beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_sparse_L1])

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_sparse_L1=ema.average(gen_loss_sparse_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        outputs_psf=outputs_psf,
        targets=targets,
        inputs=inputs,
        train=tf.group(update_losses, gen_train),
    )


