#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 17:33:10 2018

@author: useradmin
"""
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

psf_size = 31
psf_sigma = 5

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


mysize = 256 
outputs_np = 1.*np.zeros((1,mysize, mysize, 1), np.float32)
outputs_np[:,1,55,:] = 1

outputs_tf = tf.Variable(outputs_np)

# generate the heatmap corresponding to the predicted spikes
sess = tf.InteractiveSession()
init=tf.initialize_all_variables()
sess.run(init)

if(1):
    padsize = np.int8(psf_size/2)
    outputs_padded = tf.pad(outputs_tf, [[0, 0], [padsize , padsize ], [padsize , padsize ], [0, 0]], mode="REFLECT")
    outputs_psf = tf.nn.conv2d(outputs_padded, gfilter, [1, 1, 1, 1], padding="VALID")
else:
    outputs_psf = tf.nn.conv2d(outputs_tf, gfilter, strides=(1, 1, 1, 1), padding="SAME")
output_np = np.squeeze(outputs_psf.eval())
output_np = np.int8(output_np/(np.max(output_np)+1)*2**8)

print(output_np.shape)

plt.imshow(np.squeeze(output_np))
plt.imsave('test_psf.png', output_np)