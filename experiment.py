# -*- coding: utf-8 -*-
"""
Experiment

Module specifying the experimental data structures for

Long Term Behaviour Analysis of C-elegans

Experimental Data:
Shay Stern, C. Bargman Lab, The Rockefeller University 2016

"""
__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'


import os
import glob
import numpy as np

 
############################################################################
### File locations
############################################################################      

base_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/';

experiment_directory = os.path.join(base_directory, 'Experiment/Data');

data_directory = os.path.join(base_directory, 'Data')


def filename(strain = 'n2', dtype = 'xy', wid = all):
  """Returns file name for a data source"""
  
  if wid is all:
    wid = '*';
  else:
    wid = str(wid);
  
  if dtype in ['xy', 'stage', 'img']:
    fn = strain + '_' +  dtype + '_w=' + wid + '_s=all.npy';
    return os.path.join(experiment_directory, fn);
  else:
    raise RuntimeError('cannot determine data type %s' % dtype);
  

 
############################################################################
### Accessing data
############################################################################    

def load(strain = 'n2', dtype = 'xy', wid = all, stage = all, valid_only = False, replace_invalid = None, memmap = 'r'):
  fn = filename(strain = strain, dtype = dtype, wid = wid);
  if wid is all:
    wid = range(len(glob.glob(fn)));
  
  wids = np.array([wid], dtype = int).flatten();
  
  if stage is all:
    stagel = [1,2,3,4,5];
  else:
    stagel = stage;
  stagel = np.array([stagel]).flatten();
    
  #lids = label_id(label);
  
  data = np.zeros(len(wids), dtype = 'O');
  for i,w in enumerate(wids):
    data[i] = np.load(filename(strain = strain, dtype = dtype, wid = w), mmap_mode = memmap);    
    
    if stage is not all:
      sids = np.load(filename(strain = strain, dtype = 'stage', wid = w));
      sids = np.in1d(sids, stage);
    else:
      sids = np.ones(data[i].shape[0], dtype = bool);
    
    if valid_only is True:    
      sids = np.logical_and(sids, np.sum(np.isnan(data[i]), axis = 1));
    
    data[i] = data[i][sids,:];
    
    if valid_only is False and replace_invalid is not None:        
      iids = np.sum(np.isnan(data[i]), axis = 1);
      data[i][iids,:] = replace_invalid;
  
  if isinstance(wid, int):
    data = data[0];
  
  return data;


## worm images

def load_img(strain = 'n2', wid = 80, t = all):
  """Loads cropped worm images"""
  fn = filename(strain = strain, dtype = 'img', wid = wid);
  imgdata = np.load(fn, mmap_mode = 'r');
  if t is all:
    return imgdata;
  else:
    tl = np.array([t]).flatten();
    img = imgdata[tl];
    
    if isinstance(t,int):
      return img[0];
    else:
      return img;



############################################################################
### Util 
############################################################################      

def stage_switch(stage, valid = False):
  """Returns indices of developmental stage changes"""
  
  return np.argwhere(np.diff(stage, valid = valid));  



############################################################################
### Test 
############################################################################   

if __name__ == '__main__':
  fn = filename()
  print fn
  
  fn = filename(dtype = 'img')
  print fn
  print glob.glob(fn)
  
  data = load(wid = 0);
  print data.shape
  
  import matplotlib.pyplot as plt
  plt.figure(1); plt.clf();
  img = load_img(t=100000);
  plt.imshow(img, cmap = 'gray')
  
  #animate movie  
  import time

  fig, ax = plt.subplots()
  figimg = ax.imshow(img, cmap = 'gray');
  plt.show();
  
  for t in range(200000, 300000):
    figimg.set_data(load_img(t=t));
    ax.set_title('t=%d' % t);
    time.sleep(0.001)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
