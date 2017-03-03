# -*- coding: utf-8 -*-
"""
Experiment

Module specifying the experimental data structures for

long term behavioural analysis of C-elegans

Experimental data:
Shay Stern, C. Bargman Lab, The Rockefeller University 2016
"""

__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'

import os
import glob
import numpy as np

import scipy.ndimage.filters as filters 
 
############################################################################
### File locations
############################################################################      

base_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/';
"""Base Directory"""

#chekc for mounted data
if os.path.ismount('/run/media/ckirst/CElegans_N2'): # N2 data set
  base_directory = '/run/media/ckirst/CElegans_N2/CElegansBehaviour/';


data_directory = os.path.join(base_directory, 'Experiment/Data');
"""Experiment directory with the numpy data"""

raw_data_directory = os.path.join(base_directory, 'Experiment/RawData');
"""Experiment directory with the raw data"""

analysis_directory = os.path.join(base_directory, 'Analysis/WormBehaviour/Data')
"""Analysis directory for result data"""


nstages = 5;
"""Number of life stages of the worms"""

stage_names = np.array(['L1', 'L2', 'L3', 'L4', 'Adult']);
"""Names of the life stages"""


def filename(strain = 'n2', dtype = 'xy', wid = all):
  """Returns file name for a data source"""
  
  if wid is all:
    wid = '*';
  else:
    wid = str(wid);
  
  if dtype in ['xy', 'img', 'rotation', 'speed', 'roam_align', 'roam']:
    fn = strain + '_' +  dtype + '_w=' + wid + '.npy';
  elif dtype in ['roi', 'stage', 'time']:
    fn = strain + '_' + dtype + '.npy';
  else:
    raise RuntimeError('cannot determine data type %s' % dtype);
    
  if dtype == 'img':
    fn = os.path.join('Images', fn);
  
  return os.path.join(data_directory, fn);  

 
############################################################################
### Accessing data
############################################################################    

def load(strain = 'n2', dtype = 'xy', wid = all, stage = all, 
         valid_only = False, replace_invalid = None, memmap = None):
  """Loads experimental data"""
  fn = filename(strain = strain, dtype = dtype, wid = wid);
  if wid is all:
    wid = range(len(glob.glob(fn)));
  
  wids = np.array([wid], dtype = int).flatten();
  #print wids;
  
  if stage is all:
    stagel = range(nstages);
  else:
    stagel = stage;
  stagel = np.array([stagel]).flatten();
    
  #lids = label_id(label);
  
  data = np.zeros(len(wids), dtype = 'O');
  for i,w in enumerate(wids):
    data[i] = np.load(filename(strain = strain, dtype = dtype, wid = w), mmap_mode = memmap);    
    #print data[i]
    
    if stage is not all:
      sids = np.load_stage(filename(strain = strain, wid = w));
      sids = np.in1d(sids, stage);
    else:
      sids = np.ones(data[i].shape[0], dtype = bool);
    
    if valid_only is True:    
      sids = np.logical_and(sids, np.sum(np.isnan(data[i]), axis = 1));
    
    data[i] = data[i][sids];
    
    if valid_only is False and replace_invalid is not None:        
      iids = np.sum(np.isnan(data[i]), axis = 1);
      data[i][iids,:] = replace_invalid;
  
  if isinstance(wid, int):
    data = data[0];
  
  return data;


def smooth_image(img, sigma = 1.0):
  if sigma is not None:
    return filters.gaussian_filter(np.asarray(img, float), sigma);
  else:
    return img;

def load_img(strain = 'n2', wid = 0, t = all, sigma = None):
  """Loads cropped worm images"""
  
  fn = filename(strain = strain, dtype = 'img', wid = wid);
  imgdata = np.load(fn, mmap_mode = 'r');
  if t is all:
    img = imgdata;
  else:
    tl = np.array([t]).flatten();
    img = imgdata[tl];
  
  if sigma is not None:
    imgs = np.zeros(img.shape);
    for i,im in enumerate(img):
      imgs[i] = smooth_image(im, sigma = sigma);
    img = imgs;
  
  if isinstance(t,int):
    img = img[0];
  
  return img;


def load_roi(strain = 'n2', wid = all):
  fn = filename(strain = strain , wid = wid, dtype = 'roi');
  roi = np.load(fn);
  if wid is all:
    return roi;
  else:
    return roi[wid];


def load_stage_times(strain = 'n2', wid = all):
  fn = filename(strain = strain , wid = wid, dtype = 'stage');
  stage = np.load(fn);
  if wid is all:
    return stage;
  else:
    return stage[wid];


def load_stage(strain = 'n2', wid = all):
  st = load_stage_times(strain = strain, wid = wid);
  
  if wid is all:
    wids = range(st.shape[0]);
  else:
    wids = np.array([wid]).flatten()
  
  data = np.zeros(len(wids), dtype = 'O');
  for i,w in enumerate(wids):
    s  = st[w];
    ds = np.nan * np.ones(s[-1]);
    for k in range(1,len(s)):
      ds[s[k-1]:s[k]] = k;
    data[i] = ds;
  
  if isinstance(wid, int):
    data = data[0];
  
  return data;
  

def load_time(strain = 'n2', wid = all):
  fn = filename(strain = strain , wid = wid, dtype = 'time');
  times = np.load(fn);
  if wid is all:
    return times;
  else:
    return times[wid];



############################################################################
### Accessing binned data
############################################################################    

def stage_bins(strain = 'n2', wid = all, nbins_per_stage = 10):
  """Calcualte time normalized bins using nbins per stage"""
  nbins_total = nbins_per_stage * nstages;
  stage_times = load_stage_times(strain = strain);

  if wid is all:
    wids = np.arange(stage_times.shape[0], dtype = int);
  else:
    wids = np.array([wid], dtype = int).flatten();
   
  nworms = wids.shape[0];
   
  stage_times = stage_times[wids];
  stage_durations = np.diff(stage_times, axis = 1)[:,:-1];  
  
  dt = 1.0 * stage_durations / nbins_per_stage
  bins = np.zeros((nworms, nbins_total+1), dtype = int);
  for i,w in enumerate(wids):
    for s in range(nstages):
      bins[i, s*nbins_per_stage:(s+1)*nbins_per_stage] = np.asarray(np.arange(0,nbins_per_stage) * dt[i,s] + stage_times[i,s], dtype = int);
      bins[i,-1] = stage_times[i,-2];
  
  return bins;


def bin_data(data, bins, function = np.nanmean, nout = 1):
  """Bin the data according to the specified bin ranges"""
  bins_start = bins[:,:-1]; 
  bins_end   = bins[:,1:];
  nworms = data.shape[0];
  nbins = bins_start.shape[1];
  
  if nout > 1:
    binned = np.zeros((nworms, nbins, nout));
  else:
    binned = np.zeros((nworms, nbins));
  
  for w in range(nworms):
    binned[w,:] = np.array([function(data[w][s:e]) for s,e in zip(bins_start[w], bins_end[w])]);
  
  return binned;

 
def load_stage_binned(strain = 'n2', wid  = all, dtype = 'speed', nbins_per_stage = 10, function = np.nanmean, nout = 1):
  """Load data an bin at different stages"""
  if isinstance(wid, int):
    wids = [wid];
  else:
    wids = wid;
  
  data = load(strain = strain, wid = wids, dtype = dtype);
  sbins = stage_bins(strain = strain, wid = wids, nbins_per_stage = nbins_per_stage);
  bdata = bin_data(data, sbins, function = function, nout = nout);
  
  if isinstance(wid, int):
    bdata = bdata[0];
 
  return bdata;



############################################################################
### Accessing aligned data
############################################################################    

def align_data(data, start = None, end = None,  align = None, shift = None):
  """Return array with data alaigned at specified indices"""
  nworms = data.shape[0];
  if start is None:
    start = np.zeros(nworms, dtype = int);
  if end is None:
    end = np.array([len(d) for d in data], dtype = int);
  if align is None:
    align = np.zeros(nworms, dtype = int);
  if shift is None:
    shift = np.zeros(nworms, dtype = int);
  
  #deviations
  h1 = align - start;
  h2 = end - align; 
  h1max = np.max(h1);
  #h2max = np.max(h2);
  
  #aligned start and enpositions
  ast = h1max - h1 + shift;
  aed = h1max + h2 + shift;
  
  #positive start and end positions
  amin = min(np.min(ast), np.min(aed));
  amax = max(np.max(ast), np.max(aed));
  ast = ast - amin;
  aed = aed - amin;
  ntimes = amax - amin;
  
  ad = np.ones((nworms, ntimes));
  for w in range(nworms):
    ad[w,ast[w]:aed[w]] = data[w][start[w]:end[w]];
  
  return ad;

 
def load_aligned(strain = 'n2', wid = all, dtype = 'speed', align = 'L1'):
  """Load data and return as array aligned according to a stage or absolute time"""
  
  if isinstance(wid, int):
    wids = [wid];
  else:
    wids = wid;
  
  data = load(strain = strain, wid = wids, dtype = dtype);
  stage_times = load_stage_times(strain = strain, wid = wids);
  start = stage_times[:,0];
  end   = stage_times[:,nstages];
  l = np.array([len(d) for d in data]);
  end = np.min([end, l], axis = 0);
  
  if align == 'time':
    align = np.zeros(len(data), dtype = int);
    shift = load_time(strain = strain, wid = wids);
  elif isinstance(align, basestring):
    sid = np.where(stage_names == align)[0];
    if len(sid) == 0:
      raise RuntimeError('alignment %s not in %r' % (align, stage_names));
    else:
      sid = sid[0];
    align = stage_times[:, sid];
    shift = None;
  else:
    shift = None;
    
  a = align_data(data, start = start, end = end, align = align, shift = shift);
  
  if isinstance(wid, int):
    a = a[0];
  
  return a;



############################################################################
### Test 
############################################################################   

def test():
  import glob
  import analysis.experiment as exp
  reload(exp)  
  
  fn = exp.filename()
  print fn
  
  fn = filename(dtype = 'img')
  print fn
  print glob.glob(fn)
  
  data = exp.load(wid = 0);
  print data.shape
  
  import matplotlib.pyplot as plt
  plt.figure(1); plt.clf();
  img = exp.load_img(t=200000);
  plt.imshow(img, cmap = 'gray')
  
  #animate movie  
  import time

  fig, ax = plt.subplots()
  figimg = ax.imshow(img, cmap = 'gray');
  plt.show();
  
  for t in range(200000, 300000):
    figimg.set_data(exp.load_img(t=t));
    ax.set_title('t=%d' % t);
    time.sleep(0.001)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
  reload(exp)
  sbins = exp.stage_bins(wid = [0,1])
  d = exp.load_stage_binned(wid = [0,1], nbins_per_stage=10) 
  
  a = exp.load_aligned(wid = all, align='time', dtype = 'speed')  
  a_th = np.nanpercentile(a, 85);
  a[a> a_th] = a_th;
  a[np.isnan(a)] = -1.0;
  
  import analysis.plot as fplt;
  fplt.plot_array(a)
  
  a = exp.load_aligned(wid = all, align='L2', dtype = 'speed')  
  a_th = np.nanpercentile(a, 85);
  a[a> a_th] = a_th;
  a[np.isnan(a)] = -1.0;
  
  import analysis.plot as fplt;
  fplt.plot_array(a)
  