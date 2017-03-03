# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Matlab to Numpy Data Conversion Routines for Worm Images
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os, glob
import matplotlib.pyplot as plt;
import scipy.io
import numpy as np

import analysis.experiment as exp

import scripts.preprocessing.file_order as fo;
exp_names = fo.experiment_names;
dir_names = fo.directory_names;

nworms = len(exp_names)

# wid = 90 , fid = 61, '/run/media/ckirst/CElegans_N2/CElegansBehaviour/Experiment/RawData/Results290416exp/CAM814A3/shortCAM814A3CAM814_2016-04-29-171847-0061.mat'
# is corrupt!


## Convert Worm Images

for wid in range(91, nworms):
  #file_data = os.path.join(dir_names[wid], 'short%s-%s.mat' % ('%04d'));
  file_data = np.sort(np.unique(np.array(glob.glob(os.path.join(dir_names[wid], 'short*.mat')))));
  nf = len(file_data);
  
  file_save = os.path.join(exp.data_directory, 'Images/n2_img_w=%d_s=all.npy' % wid);
  
  wxy = exp.load(wid = wid);
  n = wxy.shape[0];
  
  si = 0;
  sf = 0;
  dat = scipy.io.loadmat(file_data[sf])['mydata'][0];
  
  imgs = np.zeros((n, 151, 151), dtype = 'uint8')
  
  for i in range(n):
    print 'wid: %d  step: %d/%d fid:%d,%d' % (wid, i,n, sf, si)
    img = dat[si];
    #check for format otherwise pad
    if img.ndim != 3:
      print 'image has different dimensionas: %s, step: %d/%d fid:%d,%d' % (str(img.ndim), i,n, sf, si)
      img = np.nan * np.ones((151,151,3), dtype = 'uint8');
    if img.shape != (151,151,3):
      print 'image has different shape: %s, step: %d/%d fid:%d,%d' % (str(img.shape), i,n, sf, si)
      #break;
      #median = np.median(img);
      img2 = np.nan * np.ones((151,151,3), dtype = 'uint8');
      
      img2[:min(151, img.shape[0]), :min(151, img.shape[1]), :min(3, img.shape[2])] = img[:min(151, img.shape[0]), :min(151, img.shape[1]), :min(3, img.shape[2])];
      img = img2;
    
    img  = img.sum(axis = 2)/3;
    imgs[i] = img;
    
    si += 1;
    if si == dat.shape[0]:
      sf += 1;
      if sf < nf:
        dat = scipy.io.loadmat(file_data[sf])['mydata'][0];
        si = 0;
      else:
        break;
  
  np.save(file_save, imgs)







plt.figure(1); plt.clf();
for i in range(3):
  plt.subplot(2,3,i+1);
  plt.imshow(img[:,:,i]);
plt.subplot(2,3,4);
imgt = np.sum(img, axis = 2)/3;
plt.imshow(imgt)
  
import scipy.ndimage.filters as filters 
sigma = 1.0;
imgs = filters.gaussian_filter(np.asarray(imgt, float), sigma);
plt.subplot(2,3,5);
plt.imshow(imgs)

# remove backgroung noise

plt.subplot(2,3,6);
imgf = imgs.copy();
imgf[imgf > 90] = 90;
plt.imshow(imgf)







#
#i = 18;
#fn = '/data/Science/Projects/CElegansBehaviour/Experiment/CAM819A3/corrdCAM819A3CAM819_2015-09-14-175453-%04d.mat' % i;
#data = scipy.io.loadmat(fn)
#xy = data['x_y_coor'];
#xy[:,0] = xy[:,0] - xy[0,0];
#ll = xy.shape[0];
#
#nworms = 109;
#maxcheck = 20000;
#
#dm = np.ones(nworms);
#for wid in range(nworms):
#   print 'worm %d/%d' % (wid, nworms);
#   wxy = exp.load(wid = wid, valid_only=False); 
#   d = np.zeros(maxcheck);
#   for i in range(maxcheck):
#      delta = xy[:,0] - (wxy[i:(i+ll),0] - wxy[i,0])
#      d[i] = np.abs(delta).sum();
#   dm[wid] = d.min();
#
#plt.figure(13); plt.clf(); 
#plt.plot(dm);
#plt.title( 'min=%f  n=%d' % (dm.min(), np.argmin(dm)));
#
#
#
#for off in range(len(d)):
#  xyo = xy[off:,0] - xy[off,0];
#  delta = xyo - (wxy[:xyo.shape[0],0] - wxy[0,0])
#  d[off] = np.abs(delta).sum() / delta.shape[0];
#plt.subplot(3,1,3);
#plt.plot(d)
#plt.title( 'min=%f  n=%d' % (d.min(), np.argmin(d)));
#
#plt.figure(12); plt.clf();
#
#for i in range(1,10):
#  wid = 68 + i;
#  plt.subplot(10,1,i);  
#
#  plt.plot(wxy[:10000,0]-wxy[0,0])



### Find double start experiments


dest_dir = '/run/media/ckirst/WormData1/CElegansBehaviour/Experiment/N2_Fails/Results201016/';


dir_names = fo.create_directory_names(base_directory='/run/media/ckirst/My Book/')

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



### check image size at certain time

import glob
import scipy.io

import analysis.experiment as exp


import scripts.preprocessing.file_order as fo;
exp_names = fo.experiment_names;
dir_names = fo.directory_names;

nworms = len(exp_names)


wid = 0;
img_id = 600499;

wxy = exp.load(wid = wid);
n = wxy.shape[0];

file_data = np.sort(np.unique(np.array(glob.glob(os.path.join(dir_names[wid], 'short*.mat')))));
nf = len(file_data);

si = 0;
sf = 0;
dat = scipy.io.loadmat(file_data[sf])['mydata'][0];

for i in range(n):
  print 'wid: %d  step: %d/%d fid:%d,%d' % (wid, i, n, sf, si)
  
  if i == img_id:
    img = dat[si];
    break;
  
  si += 1;
  if si == dat.shape[0]:
    sf += 1;
    if sf < nf:
      dat = scipy.io.loadmat(file_data[sf])['mydata'][0];
      si = 0;
    else:
      break;



