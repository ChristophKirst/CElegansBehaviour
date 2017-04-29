# -*- coding: utf-8 -*-
"""
Raw Data File / Directories

@author: ckirst
"""
import os
import numpy as np
import glob

import scripts.preprocessing.file_order_n2 as fo;


#%% npr-1 population


def filenames(strain = 'n2'):
  if strain == 'n2':
    exp_names = fo.experiment_names;
    dir_names = fo.directory_names;
    nworms = len(exp_names);
    # wid = 90 , fid = 61, '/run/media/ckirst/CElegans_N2/CElegansBehaviour/Experiment/RawData/Results290416exp/CAM814A3/shortCAM814A3CAM814_2016-04-29-171847-0061.mat'
    # is corrupt!
  
  elif strain == 'npr1':
    dirs = ['/run/media/ckirst/My Book/Results100316', '/run/media/ckirst/My Book/Results170316', '/run/media/ckirst/My Book/Results070915exp/'];
    dir_names = np.sort(np.hstack([np.array(glob.glob(os.path.join(d, '*'))) for d in dirs]))
    exp_names = np.array([glob.glob(os.path.join(d, '*'))[-1][-41:-9] for d in dir_names]);
    nworms = len(exp_names)
    print('npr-1: %d experiments found !' % nworms);
    
  elif strain == 'tph1':
    dirs = ['/run/media/ckirst/My Book/Results030915', '/run/media/ckirst/My Book/Results111215/'];
    dir_names = np.sort(np.hstack([np.array(glob.glob(os.path.join(d, '*'))) for d in dirs]))
    exp_names = np.array([glob.glob(os.path.join(d, '*'))[-1][-41:-9] for d in dir_names]);
    nworms = len(exp_names)
    print('tph-1: %d experiments found !' % nworms);
  
  elif strain == 'daf7':
    valid = ['207A2', '207A3', '207A4', '207A6', '800A2', '800A4', '800A5', '807A2', '807A4', '813A2', '813A3', '813A6', '819A3', '819A5'];
    dirs = ['/run/media/ckirst/My Passport/Results170815exp'];
    dir_names = np.sort(np.hstack([np.array(glob.glob(os.path.join(d, '*'))) for d in dirs]))
    dir_names = dir_names[np.array([d[-5:] in valid for d in dir_names], dtype = bool)]
    exp_names = np.array([glob.glob(os.path.join(d, '*'))[-1][-41:-9] for d in dir_names]);
    nworms = len(exp_names)
    print('daf7: %d experiments found !' % nworms);
  
  elif strain == 'cat2':
    dirs = ['/run/media/ckirst/My Passport/Results021015exp/', '/run/media/ckirst/My Passport1/Results230815exp/'];
    dir_names = np.sort(np.hstack([np.array(glob.glob(os.path.join(d, '*'))) for d in dirs]))
    exp_names = np.array([glob.glob(os.path.join(d, '*'))[-1][-41:-9] for d in dir_names]);
    nworms = len(exp_names)
    print('cat2: %d experiments found !' % nworms);

  elif strain == 'tdc1':
    dirs = ['/run/media/ckirst/My Passport1/Results270815exp/', '/run/media/ckirst/My Passport/Results031215/', '/run/media/ckirst/My Passport/Results280915exp/'];
    dir_names = np.sort(np.hstack([np.array(glob.glob(os.path.join(d, '*'))) for d in dirs]))
    exp_names = np.array([glob.glob(os.path.join(d, '*'))[-1][-41:-9] for d in dir_names]);
    nworms = len(exp_names)
    print('tdc1: %d experiments found !' % nworms);
  
  else:
    raise ValueError('no such strain %s' % strain);
    
  return nworms, exp_names, dir_names;
