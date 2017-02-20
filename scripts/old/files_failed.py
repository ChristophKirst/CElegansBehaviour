# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Clean file structure from restarted experiments
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import matplotlib.pyplot as plt;
import scipy.io
import numpy as np

import analysis.experiment as exp
exp.experiment_directory = '/run/media/ckirst/WormData1/CElegansBehaviour/Experiment/ImageData/'

import scripts.preprocessing.file_order as fo;
exp_names = fo.experiment_names;
dir_names = fo.directory_names;
#dir_names = fo.create_directory_names(base_directory='/run/media/ckirst/My Book/')

nworms = len(exp_names)


dest_dir = '/run/media/ckirst/WormData1/CElegansBehaviour/Experiment/N2_Fails/Results201016/';


import glob
import shutil as su

for i,d in enumerate(dir_names):
  l = glob.glob(os.path.join(d, 'short*0000.mat'));
  if len(l)> 1:
    ln = [os.path.split(c)[-1] for c in l]
    print i,d, ln;

    fns = glob.glob(os.path.join(d, "*%s*" % ln[0][5:37]));
    
    dd = os.path.join(dest_dir, os.path.split(d[:-1])[-1]);
    #if not os.path.exists(dd):
    #  os.mkdir(dd);
    
    fns_dest = [os.path.join(dd, os.path.split(f)[-1]) for f in fns];
    
    #for f,g in zip(fns, fns_dest):
      #print '%s -> %s' % (f,g)
      #su.move(f,g);
    
    #print fns_dest;


