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
import scipy.misc
from tifffile import imsave
# own modules
import model as model
import data as data



# training:  --mode train   --output_dir cellstorm_train  --max_epochs 1   --input_dir /home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/STORM/DATASET_NN/01_CELLPHONE_GT_PAIRS/MOV_2018_03_02_11_27_56_ISO3200_texp_1_200_RandomBlink_v5testSTORM_random_psf_v5_shifted_combined_lines_texp_1_85/train   --which_direction BtoA
# export:  --mode export   --output_dir models/cellstorm_train_AtoB_100epochs   --checkpoint cellstorm_train_AtoB_100epochs
# test: --mode test --output_dir TEST --input_dir ./Test_GAN_ISO3200  --checkpoint cellstorm_train_AtoB_100epochs
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", default='./test', help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default = 10, help="number of training epochs")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=100, help="write current training images every display_freq steps and update summaries every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--l1_sparse_weight", type=float, default=100.0, help="weight on L1 sparsity term for generator outputs")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--roi_size", type=int, default=256, help="size of the roi where the frames should be cropped out")
parser.add_argument("--x_center", type=int, default=-1, help="size of the roi where the frames should be cropped out")
parser.add_argument("--y_center", type=int, default=-1, help="size of the roi where the frames should be cropped out")
parser.add_argument("--how_many_gpu", type=int, default=1, help="how many GPUS to use? [1..2")
  
parser.add_argument("--is_tif", type=int, default=0, help="Want to save the TIF stacks on disk?")
parser.add_argument("--is_csv", type=int, default=0, help="Want to save the localization as a csv file on disk?")
parser.add_argument("--is_frc", type=int, default=0, help="Want to save two seperate tif-files for doing the FRC?")




# export options
opt = parser.parse_args()
EPS = 1e-12



# execute explicitly on GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
if(opt.how_many_gpu==2):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        
    
# initialize the random numbers
if opt.seed is None:
    opt.seed = random.randint(0, 2**31 - 1)

tf.set_random_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)


if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

if opt.mode == "test":
    if opt.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint
    options = {"ngf", "ndf", "lab_colorization"}
    with open(os.path.join(opt.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(opt, key, val)
    
for k, v in opt._get_kwargs():
    print(k, "=", v)

with open(os.path.join(opt.output_dir, "options.json"), "w") as f:
    f.write(json.dumps(vars(opt), sort_keys=True, indent=4))

# determine the sizes 
roisize = 64
scale_size = 256

# create placeholders for batchfeeding
im_xdim, im_ydim = scale_size, scale_size
inputs_tf = tf.placeholder(tf.float32, shape=(opt.batch_size, im_xdim, im_ydim, 1))
outputs_tf = tf.placeholder(tf.float32, shape=(opt.batch_size, im_xdim, im_ydim, 1))
spikes_tf = tf.placeholder(tf.float32, shape=(opt.batch_size, im_xdim, im_ydim, 1))




# inputs and targets are [batch_size, height, width, channels]
C2Pmodel = model.create_model(inputs_tf, outputs_tf, opt.ndf, opt.ngf, EPS, opt.gan_weight, opt.l1_weight, opt.l1_sparse_weight, opt.lr, opt.beta1)
#    C2Pmodel = model.create_model(examples.spikes, examples.targets, opt.ndf, opt.ngf, EPS, opt.gan_weight, opt.l1_weight, opt.l1_sparse_weight, opt.lr, opt.beta1)

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
    
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver(max_to_keep=1)

# initiate the logdir for the Tensorboard logging
logdir = opt.output_dir if (opt.trace_freq > 0 or opt.display_freq > 0) else None


# rename the output tensor
converted_outputs = tf.identity(converted_outputs, name='converted_outputs')
inputs_tf = tf.identity(inputs_tf, name='inputs_tf')

#with tf.Session() as sess:
#sess = tf.InteractiveSession()
sess = tf.Session()

#%%    Start the processing in the SESSION 

# Run the initializer
sess.run(init)

# write out the graph for later use in Android 
tf.train.write_graph(sess.graph_def, logdir,'cellstorm.pbtxt')



print("Variables have been initialized!")    
    
if opt.checkpoint is not None:
    print("loading model from checkpoint")
    checkpoint = tf.train.latest_checkpoint(opt.checkpoint)
    saver.restore(sess, checkpoint)

    print("saving model")
    saver.save(sess, os.path.join(opt.output_dir, "cellstorm.ckpt"))
  


# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

model_dir = './testdump/'
output_node_names = 'converted_outputs'

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
g = tf.GraphDef()
g.ParseFromString(open("./testdump/frozen_model.pb", "rb").read())
# check input dims
[n for n in g.node if n.name.find("Placeholder") != -1]
# check output dims
[n for n in g.node] if n.name.find("converted_outputs") != -1]
    
    
    
    

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




import argparse 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

frozen_graph_filename = './testdump/frozen_model.pb'


# We use our "load_graph" function
graph = load_graph(frozen_graph_filename)

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
x = graph.get_tensor_by_name('prefix/Placeholder:0')
y = graph.get_tensor_by_name('prefix/convert_outputs/convert_image:0')
        
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