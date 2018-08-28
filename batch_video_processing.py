#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:54:14 2018

@author: useradmin
"""

import os
import glob
import subprocess as sub


iterindex = 0
video_dir = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/STORM/Stack_6' #/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/STORM/MATLAB/Alex_Images_Vergleich/Stack_6'
for filename in glob.glob(os.path.join(video_dir , '*.m4v')):
    print('Run Data #: ', str(iterindex+1), ' / ', str(len(glob.glob(os.path.join(video_dir , '*.m4v')))), ' named: ', filename)
    
    checkpoint = 'train_overnight_1_2_3_cluster_4_GANupdaterule_synthetic'      
          
    run_cmd = 'python /media/useradmin/Data/Benedict/Dropbox/Dokumente/Promotion/PROJECTS/STORM/PYTHON/pix2pix-tensorflow/pix2pix.py --mode test --batch_size 1 --output_dir ./dump --scale_size 768 --roi_size 128 --is_csv 1 --is_tif 1 --is_frc 1 --x_center 64 --y_center 64 --max_steps 2000'
   # filename = video_dir + '/test_density_128_3SNR_Compression_nphotons_50_compression_70.m4v'
    run_cmd_video = run_cmd + ' --input_dir ' + filename 
    run_cmd_video = run_cmd_video + ' --checkpoint ' + checkpoint

    print sub.check_output(run_cmd_video, shell=True)
    
    