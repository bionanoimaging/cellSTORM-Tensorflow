from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import graph_util

from subprocess import run
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


# own modules
import model as model
import data

## define output and inputnames 
model_folder = 'cellstorm_simple_lite_4'
inputs_name = 'inputs_tf'
outputs_name = 'outputs'
batch_size = 6 # this is experimental for faster processing 


def load_graph(output_graph):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(output_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")   
    return graph

def freeze_graph(model_folder, output_node_names):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph_path = absolute_model_folder + '/frozen_model.pb'

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))

    return output_graph_path



#%% at first wie have to get back the computational graph and modify it so that it works with Android (lite)

# Load parameters from trained network
f = open(os.path.join(model_folder, "options.json"))
JObject = json.loads(f.read())
scale_size = JObject['scale_size']
gan_weight = JObject['scale_size']
lr = JObject['lr'] 
beta1 = JObject['beta1'] 
l1_sparse_weight = JObject['l1_sparse_weight'] 
l1_weight = JObject['l1_weight']
ndf = JObject['ndf']
ngf = JObject['ngf']
mode = 'test'

# specify export folder and create it
output_dir = './' + model_folder + '_exported'
try:
    os.stat(output_dir)
except:
    os.mkdir(output_dir) 
    


# determine the sizes 
roisize = 64
EPS = 1e-12
   
# create placeholders for batchfeeding
im_xdim, im_ydim = scale_size, scale_size
inputs_tf = tf.placeholder(tf.float32, shape=(batch_size, im_xdim, im_ydim, 1), name='inputs_tf')
outputs_tf = tf.placeholder(tf.float32, shape=(batch_size, im_xdim, im_ydim, 1), name='outputs_tf')
spikes_tf = tf.placeholder(tf.float32, shape=(batch_size, im_xdim, im_ydim, 1), name='spikes_tf')

# inputs and targets are [batch_size, height, width, channels]
C2Pmodel = model.create_model(inputs_tf, outputs_tf, ndf, ngf, EPS, gan_weight, l1_weight, l1_sparse_weight, lr, beta1)

# reverse any processing on images so they can be written to disk or displayed to user
outputs = data.deprocess_tf(C2Pmodel.outputs)
outputs = data.norm_min_max_tf(outputs)
outputs = tf.reduce_sum(outputs, axis=[0,3], name='outputs')
 
# define saver
saver = tf.train.Saver(max_to_keep=1)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
print("Variables have been initialized!")    

with tf.Session() as sess:
#sess = tf.InteractiveSession()
#sess = tf.Session()

    # Run the initializer
    sess.run(init)
    
    # write out the graph for later use in Android 
    tf.train.write_graph(sess.graph_def, output_dir, 'cellstorm.pbtxt')

    # first load the weights 
    checkpoint = tf.train.latest_checkpoint(model_folder)
    saver.restore(sess, checkpoint)
    
    # now safe the weights in the same directory where the graph-def is located
    saver.save(sess, os.path.join(output_dir, "cellstorm.ckpt"))

# freeze the graph
output_graph_path = freeze_graph(output_dir, outputs_name)

#%% Here we start converting the graph to Anrdoid Format
# We use our "load_graph" function
mygraph = load_graph(output_graph_path)

# We can verify that we can access the list of operations in the graph
# for op in mygraph.get_operations():
    # print(op.name)
    # prefix/Placeholder/inputs_placehoutslder
    # ...
    # prefix/Accuracy/predictions
        

#%% We launch a Session
 # We access the input and output nodes 
node_input = mygraph.get_tensor_by_name('prefix/'+inputs_name +':0')
node_output = mygraph.get_tensor_by_name('prefix/'+outputs_name+':0')

with tf.Session(graph=mygraph) as sess:
    #simulate network with some data
    
    if(0):
        # Test if localizing from random data works works
        randomstorm = np.random.randn(batch_size,256,256,1)
        randomstorm = randomstorm* (randomstorm > 0.9)
        
        for i_image in range(0, batch_size):
            randomstorm[i_image,:,:,:] = gaussian_filter(randomstorm[i_image,:,:,:], sigma=9) 
        #
        randomstorm = randomstorm-np.min(randomstorm)
        randomstorm = randomstorm/np.max(randomstorm)
        randomstorm =  data.preprocess(randomstorm)
    elif(0):
        # Test if localizing from a TIFF works
        mytiffile_name = 'test_if_it_works.tif'
        import tifffile as tif
        import scipy.misc
        
        randomstorm = np.zeros((batch_size,256,256))
        
        for i_image in range(0, batch_size):
            myframe = tif.imread(mytiffile_name, key=i_image)
            myframe = myframe/np.max(myframe)
            myframe = data.preprocess(myframe)
            # resize to scale_size
            
            myframe = scipy.misc.imresize(myframe, size = (256, 256), interp='bilinear', mode='F')
            randomstorm[i_image,:,:] = myframe

        randomstorm = np.expand_dims(randomstorm, axis=3)
        
    else:
        # Test if localizing from a video works
        roisize = 64
        scale_size = 256
        input_dir = '/home/diederich/Documents/STORM/DATASET_NN/All_Videos/2017-12-18_18.29.45.mp4'
        VideoReader = data.VideoReader(input_dir, scale_size, roisize, -1, -1)
        VideoReader.select_ROI() # select the ROI coordinates
        
        randomstorm = np.zeros((batch_size,256,256))
        for i_image in range(0, batch_size):
            _, myframe = VideoReader.__getitem__(100+i_image)
            randomstorm[i_image,:,:] = np.squeeze(myframe)
            
        # add dimension to simulate batch    
        randomstorm = np.expand_dims(randomstorm, axis=3)  
    #randomstorm = 0.6*(randomstorm*2-1)
    

    y_out = sess.run(node_output, feed_dict={node_input: randomstorm})
    #
    for i_image in range(0, batch_size):
        plt.imshow(np.squeeze((randomstorm[i_image,:,:,:])), cmap='gray'), plt.colorbar()
        plt.show()
        plt.imshow(np.squeeze(y_out), cmap='gray'), plt.colorbar()
        plt.show()    
        
   

#%% Optimize graph	        
inputGraph = tf.GraphDef()
with tf.gfile.Open(output_graph_path, "rb") as f:
    data2read = f.read()
    inputGraph.ParseFromString(data2read)

outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                [inputs_name], # an array of the input node(s)
                [outputs_name], # an array of output nodes
                tf.int32.as_datatype_enum)

        # Save the optimized graph

output_graph_path_opt = './'+model_folder+'/'+model_folder+'_opt.pb'
f = tf.gfile.FastGFile(output_graph_path_opt, "w")
f.write(outputGraph.SerializeToString())    

    

#%%freeze_graph(checkpoint_name, outputs_name)

def bash(command):
    run(command.split())
    
    
bash('toco \
--graph_def_file=./cellstorm_simple_lite_7/cellstorm_simple_lite_4.pb \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_type=FLOAT \
--input_type=FLOAT \
--input_arrays=inputs_tf \
--output_arrays=generator/outputs \
--input_shapes=1,256,256,1 \
--output_file=./cellstorm_simple_lite_7/cellstorm_simple_lite_4.tflite')

