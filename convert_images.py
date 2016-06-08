# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Matlab to Numpy Data Conversion Routines for Worm Images
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

# check cam data:
import matplotlib.pyplot as plt;
import scipy.io
import np


import experiment as exp
reload(exp)

wxy = exp.load(wid = 96, valid_only=False);


i = 0;
i = 2;

i += 1;
#fn = '/data/Science/Projects/CElegansBehaviour/Experiment/CAM814A4/corrdCAM814A4CAM814_2015-11-20-174505-%04d.mat' % i;
fn = '/data/Science/Projects/CElegansBehaviour/Experiment/CAM819A3/corrdCAM819A3CAM819_2015-09-14-175453-%04d.mat' % i;

data = scipy.io.loadmat(fn)
xy = data['x_y_coor']

plt.figure(10); plt.clf();
plt.subplot(3,1,1);
plt.plot(xy[:,0] - xy[0,0])
plt.subplot(3,1,2);
plt.plot(wxy[:(xy.shape[0]),0]- wxy[0,0])
plt.title(str(i));


# test offset:
d = np.zeros(xy.shape[0]- 10);
for off in range(len(d)):
  xyo = xy[off:,0] - xy[off,0];
  delta = xyo - (wxy[:xyo.shape[0],0] - wxy[0,0])
  d[off] = np.abs(delta).sum() / delta.shape[0];
plt.subplot(3,1,3);
plt.plot(d)
plt.title( 'min=%f  n=%d' % (d.min(), np.argmin(d)));

## CAM814A4 -> i = 22, shift = 561 for wid = 80
## CAM819A3 -> i = 22, shift = 135 wor wid = 96

### Convert data sets into numyp arrays to match xy data

## Convert Worm Images

wids = [80, 96];

startfile = [22,22];
startindex = [561, 135];

fileformat = ['/data/Science/Projects/CElegansBehaviour/Experiment/CAM814A4/shortCAM814A4CAM814_2015-11-20-174505-%04d.mat',
              '/data/Science/Projects/CElegansBehaviour/Experiment/CAM819A3/shortCAM819A3CAM819_2015-09-14-175453-%04d.mat'];

filesave = ['/data/Science/Projects/CElegansBehaviour/Experiment/Data/Images/n2_img_wid=80.npy',
            '/data/Science/Projects/CElegansBehaviour/Experiment/Data/Images/n2_img_wid=96.npy']

# find the number of data points

idx = 1;
wid = wids[idx];
sf = startfile[idx];
si = startindex[idx];
ff = fileformat[idx];
fs = filesave[idx];

wxy = exp.load(wid = wid);
n = wxy.shape[0];

dat = scipy.io.loadmat(ff % sf)['mydata'][0];

imgs = np.zeros((n, 151, 151), dtype = 'uint8')
for i in range(n):
  print 'step: %d/%d fid:%d,%d' % (i,n, sf, si)
  img = dat[si];
  #check for format otherwise pad
  if img.ndim != 3:
    print 'image has different dimensionas: %s, step: %d/%d fid:%d,%d' % (str(img.ndim), i,n, sf, si)
    img = np.zeros((151,151,3), dtype = 'uint8');
  if img.shape != (151,151,3):
    print 'image has different shape: %s, step: %d/%d fid:%d,%d' % (str(img.shape), i,n, sf, si)
    img2 = np.zeros((151,151,3), dtype = 'uint8');
    img2[:min(151, img.shape[0]), :min(151, img.shape[1]), :min(3, img.shape[2])] = img[:min(151, img.shape[0]), :min(151, img.shape[1]), :min(3, img.shape[2])];
    img = img2;
  
  img  = img.sum(axis = 2)/3;
  imgs[i] = img;
  
  si += 1;
  if si == dat.shape[0]:
    sf+=1;
    dat = scipy.io.loadmat(ff % sf)['mydata'][0];
    si = 0;

np.save(fs, imgs)







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
