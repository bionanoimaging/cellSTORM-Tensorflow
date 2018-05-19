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

parser.add_argument("--is_video", type=int, default=0, help="input for testing is a video-file?")
parser.add_argument("--roi_size", type=int, default=256, help="size of the roi where the frames should be cropped out")
parser.add_argument("--x_center", type=int, default=256, help="size of the roi where the frames should be cropped out")
parser.add_argument("--y_center", type=int, default=256, help="size of the roi where the frames should be cropped out")
parser.add_argument("--how_many_gpu", type=int, default=1, help="how many GPUS to use? [1..2")
  



# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
opt = parser.parse_args()

EPS = 1e-12



# execute explicitly on GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
if(opt.how_many_gpu==2):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        
    



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

print('Load the data - could take a while!')    
# chooose what kind of input data to choose
if opt.is_video:
    # create Video-Reader 
    roisize = opt.roi_size
    xcenter, ycenter = 1080/2, 1920/2
    VideoReader = data.VideoReader(opt.input_dir, opt.scale_size, roisize, xcenter, ycenter)
    examples = VideoReader.loadDummy() # bad workaround

else:
    if(0):
        examples = data.load_examples(opt.input_dir, opt.scale_size, opt.batch_size, opt.mode)
    else:
        # define the path to the datafiles
        data_file_1 = './data/gt_100k_density_1.csv.h5'
        data_file_2 = './data/2017-12-18_18.29.45.mp4_256_Tirf_v2_from_video_v2_fakeB_2k.csv.h5'
        data_file_3 = './data/MOV_2018_04_20_16_31_44_Sample_Larve_ISO1200_texp1_60.csv.h5'
        
        # load each set seperatly 
        print('Load Dataset #1')
        examples_1 = data.load_examples_h5(data_file_1, batch_size=opt.batch_size)
        print('Load Dataset #2')
        examples_2 = data.load_examples_h5(data_file_2, batch_size=opt.batch_size)
        print('Load Dataset #3')
        examples_3 = data.load_examples_h5(data_file_3, batch_size=opt.batch_size)
        
        # merge the examples, but only put the same number of samples from each dataset! 
        # => avoids predomination of one specific dataset
        # get num of samples take the min of all 
        num_samples = np.int32(np.min((examples_1.inputs.shape[0], examples_2.inputs.shape[0], examples_3.inputs.shape[0])))
        print('Number of samples for each Dataset: ', num_samples)
        
        # limmit the number of samples to the same number 
        examples_1 = data.limit_numsamples(examples_1, num_samples)
        examples_2 = data.limit_numsamples(examples_2, num_samples)
        examples_3 = data.limit_numsamples(examples_3, num_samples)
        print('Merge Datasets')
        examples_12 = data.merge_examples(examples_1, examples_2)
        examples = data.merge_examples(examples_12, examples_3)

       
        # join all variables from all datasets together
        all_inputs = examples.inputs
        all_targets = examples.targets
        all_spikes = examples.spikes
        max_steps = examples.inputs.shape[0]

        # shuffle along first dimeions = numsaples
        from sklearn.utils import shuffle
        all_inputs, all_targets, all_spikes = shuffle(all_inputs, all_targets, all_spikes, random_state=0)
        
   
print('Data-loading complete!')     
    

# create placeholders for batchfeeding
im_xdim, im_ydim = examples.inputs.shape[1], examples.inputs.shape[2]
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


#%% summaries
with tf.name_scope("inputs_summary"):
    tf.summary.image("inputs", converted_inputs)

with tf.name_scope("targets_summary"):
    tf.summary.image("targets", converted_targets)

with tf.name_scope("outputs_summary"):
    tf.summary.image("outputs", converted_outputs)

with tf.name_scope("outputspsf_summary"):
    tf.summary.image("outputs_psf", converted_outputs_psf)

with tf.name_scope("predict_real_summary"):
    tf.summary.image("predict_real", tf.image.convert_image_dtype(C2Pmodel.predict_real, dtype=tf.uint8))

with tf.name_scope("predict_fake_summary"):
    tf.summary.image("predict_fake", tf.image.convert_image_dtype(C2Pmodel.predict_fake, dtype=tf.uint8))

tf.summary.scalar("discriminator_loss", C2Pmodel.discrim_loss)
tf.summary.scalar("generator_loss_GAN", C2Pmodel.gen_loss_GAN)
tf.summary.scalar("generator_loss_L1", C2Pmodel.gen_loss_L1)
tf.summary.scalar("generator_loss_sparse_L1", C2Pmodel.gen_loss_sparse_L1)


# add histogramm summary for all trainable values
for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

# add histogramm summary for gradients
for grad, var in C2Pmodel.discrim_grads_and_vars + C2Pmodel.gen_grads_and_vars:
    tf.summary.histogram(var.op.name + "/gradients", grad)

with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1)

# initiate the logdir for the Tensorboard logging
logdir = opt.output_dir if (opt.trace_freq > 0 or opt.display_freq > 0) else None



#with tf.Session() as sess:
sess = tf.InteractiveSession()
#%%    Start the processing in the SESSION 

if(1):   
    # Run the initializer
    sess.run(init)
    
    print("parameter_count =", sess.run(parameter_count))

    
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    
    
    print("Variables have been initialized!")    
        
    if opt.checkpoint is not None:
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(opt.checkpoint)
        saver.restore(sess, checkpoint)

    
    if opt.mode == "test":
        # testing
        # at most, process the test data once
        start = time.time()
        experiment_name = opt.input_dir.split("/")[-1]
        network_name =  opt.checkpoint
        
        if opt.is_video:
            max_steps = VideoReader.__len__()
        
        for step_i in range(0, max_steps, opt.batch_size):
            
            step = np.mod(step_i, examples.count)
            epoch = np.floor_divide(step_i, examples.count)
            # if a video is selected read frames instead of images directly from disk
            if opt.is_video == True:
				
                inputframe_raw, input_frame_processed = VideoReader.__getitem__(step)
            
                
                inputs_np, outputs_np, outputs_psf_np = sess.run([inputs, outputs, outputs_psf], feed_dict= {inputs_tf : input_frame_processed})
            
                # hacky workaround to keep model as is
                outputs_np = np.squeeze(np.array(outputs_np))
                inputs_np =  np.squeeze(inputframe_raw)
                outputs_psf_np = np.squeeze(np.array(outputs_psf_np))
                
                # Deprocess bring back to 0..1
                if(0):
                    outputs_np = (outputs_np + 1) / 2
                    inputs_np = (inputs_np + 1) / 2
                    outputs_psf_np = (outputs_psf_np + 1) / 2
                if(1):
                    # deprocess the data -1..1 -> 0..1
                    outputs_np = data.deprocess(outputs_np)
                    outputs_psf_np = data.deprocess(outputs_psf_np)
                    
                if(1):    
                    # Convert Back to uint8
                    outputs_np = np.uint8(2**8*outputs_np)
                    outputs_psf_np = np.uint8(2**8*outputs_psf_np)
                     
                
                # save frames to TIF 
                data.save_as_tif(inputs_np, outputs_np, outputs_psf_np, experiment_name, network_name)
            
            print("evaluated image " + str(step))
            
                
        print("rate", (time.time() - start) / max_steps)
    else:
        # training
        start = time.time()
        

        
        
        
    
        for epoch_i in range(opt.max_epochs):

            # calculate randomized selection of samples at training time 
            iter_batches = np.int32((max_steps-opt.batch_size)/opt.batch_size)
            iter_index = opt.batch_size*np.arange(iter_batches)
            np.random.shuffle(iter_index) # shuffle indices 
            
            for step in range(max_steps-opt.batch_size):
                

                
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
                
                
                # evaluate result for one frame at a time
                # get the slices from the HDF5 data-stack:
                index_i = iter_index[np.int32(step/4)]
                input_np = all_inputs[index_i:index_i+opt.batch_size,:,:,:]
                targets_np = all_targets[index_i:index_i+opt.batch_size,:,:,:]
                spikes_np = all_spikes[index_i:index_i+opt.batch_size,:,:,:]
                
                # reduce the cost-function
                sess.run(C2Pmodel.train, feed_dict= {inputs_tf:input_np, outputs_tf:spikes_np})       
        
        
                if should(opt.progress_freq):
                    discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_sparse_L1 = sess.run([C2Pmodel.discrim_loss, C2Pmodel.gen_loss_GAN, C2Pmodel.gen_loss_L1, C2Pmodel.gen_loss_sparse_L1])
    
                if should(opt.display_freq):
                    merged_summary_op = tf.summary.merge_all()
                    # Write logs at every iteration
                    
                    print("recording summary")
                    summary, nputs_np, outputs_np, outputs_psf_np = sess.run([merged_summary_op, inputs, outputs, outputs_psf], feed_dict= {inputs_tf:input_np, outputs_tf:targets_np})
                        
                    # add all summaries to the writer
                    # Merge all summaries into a single op            
                    summary_writer.add_summary(summary, step)
        
               
    
                    
    
    
                    
        
        
        
                if should(opt.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    print("Current step in data: ", step, ", epoch: ", epoch_i)
                    print("progress  step %d  "% step)
                    print("discrim_loss", discrim_loss)
                    print("gen_loss_GAN", gen_loss_GAN)
                    print("gen_loss_L1", gen_loss_L1)
                    print("gen_loss_sparse_L1", gen_loss_sparse_L1)
        
                if should(opt.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(opt.output_dir, "C2Pmodel"), global_step=step)
        
               
