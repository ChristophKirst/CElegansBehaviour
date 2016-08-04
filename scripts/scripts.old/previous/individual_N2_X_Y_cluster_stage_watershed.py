# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:35:50 2016

@author: ckirst
"""

  
import os
import numpy as np
import scipy.io as io


import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt


from wormdata import shiftData

import wormdata as wd;
reload(wd)

basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/';
filename = os.path.join(basedir, 'Experiment/individuals_N2_X_Y.mat')

data = io.loadmat(filename);
XYdata = data['individual_X_Y'][0];

print XYdata.shape
print XYdata[0].shape  

nworms = XYdata.shape[0];

stages = [1,2,3,4,5];
#stages = np.unique(w.stage());

datadir = os.path.join(basedir, 'Analysis/Data/2016_05_23_Classification')
figdir  = os.path.join(datadir, 'Figures');


delays = [0, 5, 10, 30, 100, 150, 200, 300, 400, 500];
delays_max = max(delays)+1
n_delays = len(delays);
n2_delays = int(np.ceil(n_delays/2.))

features = ['distances', 'rotations'];



### Clustering using s=4 d=10


from utils import smooth
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import matplotlib.cm as cm;

d = 3;
s = 1;
nbins = 256;


def makeSegmentation(d, s, nbins = 256, verbose = True, sigma = 0.75, min_distance = 1):
  fn = os.path.join(datadir, 'data_stage%d_%s.npy' % (s,features[0]));
  dists = np.load(fn);
  fn = os.path.join(datadir, 'data_stage%d_%s.npy' % (s,features[1]));
  rots = np.load(fn);
  
  ddata = np.vstack([np.log(dists[:,d]), (rots[:,d])]).T
  #gmmdata = np.vstack([dists[:,j], (rots[:,j])]).T
  #ddata.shape
  
  nanids = np.logical_or(np.any(np.isnan(ddata), axis=1), np.any(np.isinf(ddata), axis=1));
  ddata = ddata[~nanids,:];
  #ddata.shape
  imgbin = None;
  img2 = smooth(ddata, nbins = [nbins, nbins], sigma = (sigma,sigma))
  #img = smooth(ddata, nbins = [nbins, nbins], sigma = (1,1))

  local_maxi = peak_local_max(img2, indices=False, min_distance = min_distance)
  imgm2 = img2.copy();
  imgm2[local_maxi] = 3 * imgm2.max();
  
  if verbose:
    imgbin = smooth(ddata, nbins = [nbins, nbins], sigma = None)
    plt.figure(220); plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(imgbin)
    plt.subplot(2,2,2)
    plt.imshow(img2)
    plt.subplot(2,2,3);
    plt.imshow(imgm2,  cmap=plt.cm.jet, interpolation='nearest')

  markers = ndi.label(local_maxi)[0]
  labels = watershed(-img2, markers, mask = None);
  print "max labels: %d" % labels.max()

  if verbose:
    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax0, ax1, ax2 = axes
    ax0.imshow(img2, cmap=plt.cm.jet, interpolation='nearest')
    ax0.set_title('PDF')
    labels[imgbin==0] = 0;
    labels[0,0] = -1;
    ax1.imshow(labels, cmap=plt.cm.rainbow, interpolation='nearest')
    ax1.set_title('Segmentation on Data')
    #labelsws[0,0] = -1;
    #ax2.imshow(labelsws, cmap=plt.cm.rainbow, interpolation='nearest')
    #ax1.set_title('Segmentation Full')
    
  return labels;


#classification for a specific work based on the segmentation above
def classifyWorm(labels, wid, d, s, nbins = 256, verbose = True):
  XYwdata = XYdata[wid].copy();
  w = wd.WormData(XYwdata[:,0:2], stage = XYwdata[:,-1], valid = XYwdata[:,0] != 1, label = ('x', 'y'), wid = wid);
  w.replaceInvalid(); 
  
  ds = w.calculateDistances(n = delays[d]+1, stage = s); 
  rs = w.calculateRotations(n = delays[d]+1, stage = s); 
  
  ddata = np.vstack([np.log(ds[:,-1]), (rs[:,-1])]).T
  #gmmdata = np.vstack([dists[:,j], (rots[:,j])]).T
  #ddata.shape
  
  nanids = np.logical_or(np.any(np.isnan(ddata), axis=1), np.any(np.isinf(ddata), axis = 1));
  ddata = ddata[~nanids,:];
  #ddata.shape
  
  pred2 =-np.ones(rs.shape[0])
  pred2nn = pred2[~nanids];
  pred2nn.shape
  
  for i in range(2):
    ddata[:,i] = ddata[:,i] - ddata[:,i].min();
    ddata[:,i] = (ddata[:,i] / ddata[:,i].max()) * (nbins-1);
    
  ddata = np.asarray(ddata, dtype = int);
    
  for i in range(2):
    ddata[ddata[:,i] > (nbins-1), i] = nbins-1;
  
  for i in xrange(ddata.shape[0]):
    pred2nn[i] = labels[ddata[i,0], ddata[i,1]];
    
  pred2[~nanids] = pred2nn;
  #pred2nn.max();
  
  if verbose:
    plt.figure(506); plt.clf();
    w.plotTrace(ids = shiftData(pred2, delays[d]/2, nan = -1), stage = s)
  
    if verbose > 2:
      rds = w.calculateRotations(n = delays[d] + 1, stage = s);
      plt.figure(510); plt.clf();
      w.plotDataColor(data = shiftData(rds[:, -1], delays[d]/2, nan = -1), c = pred2, lw = 0, s = 20, stage = s, cmap = cm.rainbow)
    
      dts = w.calculateDistances(n = delays[d] + 1, stage = s);
      plt.figure(510); plt.clf();
      w.plotDataColor(data = dts[:, -1], c = pred2, lw = 0, s = 20, stage = s, cmap = cm.rainbow)
    
      plt.figure(507); plt.clf();
      w.plotTrajectory(stage = s, colordata = shiftData(rds[:,-1] /(delays[d]+1) /np.pi, delays[d]/2, nan = -.1))
    
      dist2 = w.calculateLengths(n=200);
      plt.figure(511); plt.clf();
      w.plotDataColor(data = dist2[:, -1], c = pred2, lw = 0, s = 20)
  
  return pred2;


#assume classes to be 0...N
def calcRates(classpred, ):
  classes = np.unique(classpred);
  nclasses = len(classes);
  
  #map classes to array index:
  transitions = np.zeros((nclasses, nclasses));
  for i in range(classpred.shape[0]-1):
    transitions[classpred[i], classpred[i+1]] += 1;
    
  return transitions
    

d = 3;
s = 1;
nbins = 256;
verbose = False;

labels = makeSegmentation(d = d, s = s, nbins = nbins, verbose = verbose);

rates = [];
for wid in range(50):
  print "worm %d" % wid
  classes = classifyWorm(labels = labels, wid = wid, d= d, s = s, nbins = nbins, verbose = verbose);

  classes[classes == -1] = 0;
  rates.append(calcRates(classes))
  
  
plt.figure(400); plt.clf();
for i,r in enumerate(rates):
  plt.subplot(1, len(rates), i+1);
  plt.imshow(np.log(r[1:,1:]), interpolation='nearest')

plt.figure(15); plt.clf();
for i in range(2):
  for j in range(2):
    plt.plot([r[i,j] for r in rates])
  




d = 3;
s = 4;
nbins = 256;
verbose = False;

labels = makeSegmentation(d = d, s = s, nbins = nbins, verbose = verbose, sigma = 1, min_distance = 5);

rates = [];
for wid in range(50):
  print "worm %d" % wid
  classes = classifyWorm(labels = labels, wid = wid, d= d, s = s, nbins = nbins, verbose = verbose);

  classes[classes == -1] = 0;
  rates.append(calcRates(classes))
  
  
plt.figure(400+s); plt.clf();
for i,r in enumerate(rates):
  plt.subplot(1, len(rates), i+1);
  plt.imshow(np.log(r[1:,1:]), interpolation='nearest')

plt.figure(15+s); plt.clf();
ns = len(rates[0]);
for i in range(ns):
  for j in range(ns):
    plt.plot([r[i,j] for r in rates])
    
    
    
    
d = 2;
s = 4;
nbins = 256;
verbose = False;

labels = makeSegmentation(d = d, s = s, nbins = nbins, verbose = True, sigma = 1, min_distance = 5);

wid = 60;
classifyWorm(labels = labels, wid = wid, d= d, s = s, nbins = nbins, verbose = True);


rates = [];
for wid in range(50):
  print "worm %d" % wid
  classes = classifyWorm(labels = labels, wid = wid, d= d, s = s, nbins = nbins, verbose = verbose);

  classes[classes == -1] = 0;
  rates.append(calcRates(classes))
  
  
plt.figure(400+s); plt.clf();
for i,r in enumerate(rates):
  plt.subplot(1, len(rates), i+1);
  plt.imshow(np.log(r[1:,1:]), interpolation='nearest')

plt.figure(15+s); plt.clf();
ns = len(rates[0]);
for i in range(ns):
  for j in range(ns):
    plt.plot([r[i,j] for r in rates])



