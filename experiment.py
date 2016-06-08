# -*- coding: utf-8 -*-
"""
Experiment

Module specifying the experimental data structures for 

Long Term Behaviour Analysis of C-elegans

Experimental Data:
Shay Stern, Cori Bargman Lab, The Rockefeller University 2016

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
  
  if dtype == 'xy':
    fn = strain + '_' +  dtype + '_wid=' + wid + '.npy';
    return os.path.join(experiment_directory, fn);
  elif dtype == 'img':
    fn = strain + '_' +  dtype + '_wid=' + wid + '.npy';
    return os.path.join(experiment_directory, 'Images/' + fn);  
  else:
    raise RuntimeError('cannot determine data type %s' % dtype);
  

 
############################################################################
### Data structure
############################################################################    

## XY data

labelpos= {'x' : 0, 'y' : 1, 'stage' : 2};

def label_string(self, i = all):
  """Returns string label of the data"""
  if i is all:
    return labelpos.keys();
  else:
    if isinstance(i, int):
      for (k,v) in labelpos.items():
        if i in np.array(v).flatten():
          return k;
    else:
      l = [];
      for (k,v) in labelpos.items():
        for ii in i:
          if i in np.array(v).flatten():
            l.append(k);
      return l;
      
def label_id(self, label = all):
  """Return label ids of label given as strings or ids"""
  
  def makelab(lab):
    if isinstance(lab, str):
      return labelpos[lab];
    else:
      return lab;
  
  if label is all:
    #return range(len(labeldict));
    return range(2); # dont want stage indices 
  else:
    if isinstance(label, str) or isinstance(label, int):
      return makelab(label);
    else:
      return np.array([makelab(l) for l in label]).flatten();


def load(strain = 'n2', dtype = 'xy', wid = all, stage = all, label = all, valid_only = False, replace_invalid = None, memmap = 'r'):
  fn = filename(strain = strain, dtype = dtype, wid = wid);
  if wid is all:
    wid = range(len(glob.glob(fn)));
  
  wids = np.array([wid], dtype = int).flatten();
  
  if stage is all:
    stage = [1,2,3,4,5];
  stage = np.array([stage]).flatten();
    
  lids = label_id(label);
  
  data = np.zeros(len(wids), dtype = 'O');
  for i,w in enumerate(wids):
    data[i] = np.load(filename(strain = strain, dtype = dtype, wid = w), mmap_mode = memmap);    
    
    sids =  np.in1d(data[i][:, labelpos['stage']], stage);
    if valid_only is True:    
      sids = np.logical_and(sids, data[i][:, labelpos['x']] != np.nan);
    data[i] = data[i][sids,:][:,lids];
    
    if valid_only is False and replace_invalid is not None:        
      iids = data[i][:, labelpos['x']] == np.nan;
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
    
