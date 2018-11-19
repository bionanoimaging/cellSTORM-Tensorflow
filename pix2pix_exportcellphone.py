from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json

import matplotlib.pyplot as plt

# own modules
import model as model
import data as data




# define location of checkpoint  (only parameter to set here!)
checkpoint = 'cellstorm_simple'

# Load parameters from trained network
options = {"ngf", "ndf", "lab_colorization"}
f = open(os.path.join(checkpoint, "options.json"))
JObject = json.loads(f.read())

mode = 'test'
output_dir = './testdump'
input_dir = 'cellstorm_simple_3layers'
batch_size = 1

scale_size = JObject['scale_size']
gan_weight = JObject['scale_size']
lr = JObject['lr'] 
beta1 = JObject['beta1'] 
l1_sparse_weight = JObject['l1_sparse_weight'] 
l1_weight = JObject['l1_weight']
ndf = JObject['ndf']
ngf = JObject['ngf']
 
# determine the sizes 
roisize = 64
scale_size = 256
EPS = 1e-12

# execute explicitly on GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        


if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# create placeholders for batchfeeding
im_xdim, im_ydim = scale_size, scale_size
inputs_tf = tf.placeholder(tf.float32, shape=(batch_size, im_xdim, im_ydim, 1), name='inputs_tf')
outputs_tf = tf.placeholder(tf.float32, shape=(batch_size, im_xdim, im_ydim, 1), name='outputs_tf')
spikes_tf = tf.placeholder(tf.float32, shape=(batch_size, im_xdim, im_ydim, 1), name='spikes_tf')




# inputs and targets are [batch_size, height, width, channels]
C2Pmodel = model.create_model(inputs_tf, outputs_tf, ndf, ngf, EPS, gan_weight, l1_weight, l1_sparse_weight, lr, beta1)
#    C2Pmodel = model.create_model(examples.spikes, examples.targets, ndf, ngf, EPS, gan_weight, l1_weight, l1_sparse_weight, lr, beta1)

 # reverse any processing on images so they can be written to disk or displayed to user
inputs = data.deprocess_tf(C2Pmodel.inputs)
targets = data.deprocess_tf(C2Pmodel.targets)
outputs = data.deprocess_tf(C2Pmodel.outputs)
outputs_psf = data.deprocess_tf(C2Pmodel.outputs_psf)

def convert(image):
    image = data.norm_min_max_tf(image)
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

with tf.name_scope("convert_inputs"):
    converted_inputs = convert(inputs)

with tf.name_scope("convert_targets"):
    converted_targets = convert(targets)

with tf.name_scope("convert_outputs"):
    converted_outputs = convert(outputs)

with tf.name_scope("convert_outputspsf"):
    converted_outputs_psf = convert(outputs_psf)


# rename the output tensor
converted_outputs = tf.identity(converted_outputs, name='converted_outputs')
outputs = tf.identity(outputs, name='outputs')
inputs_tf = tf.identity(inputs_tf, name='inputs_tf')
    
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver(max_to_keep=1)

# initiate the logdir for the Tensorboard logging
logdir = output_dir 

#with tf.Session() as sess:
#sess = tf.InteractiveSession()
sess = tf.Session()

#%%    Start the processing in the SESSION 

# Run the initializer
sess.run(init)

# write out the graph for later use in Android 
tf.train.write_graph(sess.graph_def, logdir,'cellstorm.pbtxt')



print("Variables have been initialized!")    
    
if checkpoint is not None:
    print("loading model from checkpoint")
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    saver.restore(sess, checkpoint)

    print("saving model")
    saver.save(sess, os.path.join(output_dir, "cellstorm.ckpt"))
  


# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

model_dir = './testdump/'
output_node_names = 'outputs'

#freeze_graph(model_dir, output_node_names)

# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_dir + "/frozen_model.pb"

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True

# We start a session using a temporary fresh Graph
with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))



### Check if everythings fine with the graph
import tensorflow as tf
frozen_graph_filename = './testdump/frozen_model.pb'
g = tf.GraphDef()
g.ParseFromString(open(frozen_graph_filename, "rb").read())
# check input dims
[n for n in g.node if n.name.find("inputs") != -1]
# check output dims
[n for n in g.node if n.name.find("deprocess_2") != -1]
    
    
    
    

import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")   
    return graph


#%% Here we start converting the graph to Anrdoid Format


# We use our "load_graph" function
graph = load_graph(frozen_graph_filename)

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)
    # prefix/Placeholder/inputs_placehoutslder
    # ...
    # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
x = graph.get_tensor_by_name('prefix/inputs_tf:0')
y = graph.get_tensor_by_name('prefix/outputs:0')
        
# We launch a Session
with tf.Session(graph=graph) as sess:
    # Note: we don't nee to initialize/restore anything
    # There is no Variables in this graph, only hardcoded constants 
    randomstorm = np.random.randn(1,256,256,1)
    randomstorm = randomstorm-np.min(randomstorm)
    randomstorm = randomstorm/np.max(randomstorm)
    randomstorm = randomstorm* (randomstorm > 0.9)
    plt.imshow(np.squeeze(randomstorm))
    plt.show()
    y_out = sess.run(y, feed_dict={x: randomstorm})
    # I taught a neural net to recognise when a sum of numbers is bigger than 45
    plt.imshow(np.squeeze(y_out))
    plt.show()    
    

##### Optimize graph	
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

inputGraph = tf.GraphDef()
with tf.gfile.Open(frozen_graph_filename, "rb") as f:
    data2read = f.read()
    inputGraph.ParseFromString(data2read)

outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                ['inputs_tf'], # an array of the input node(s)
                ['outputs'], # an array of output nodes
                tf.int32.as_datatype_enum)

        # Save the optimized graph

f = tf.gfile.FastGFile("outputOptimizedGraph.pb", "w")
f.write(outputGraph.SerializeToString())    

    

#%%freeze_graph(model_dir, output_node_names)
from subprocess import run

def bash(command):
    run(command.split())
    
    
bash('toco \
--graph_def_file=./testdump/frozen_model.pb \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_type=FLOAT \
--input_type=FLOAT \
--input_arrays=inputs_tf \
--output_arrays=outputs \
--input_shapes=1,256,256,1 \
--output_file=./testdump/frozen_model.tflite')

