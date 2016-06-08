"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Clustering / Feature Detection / Classification of Behaviour

Generation of PDFs, Behavioural motives and Ethographs

Experimental Data: 
Shay Stern, Cori Bargman lab, The Rockefeller University 2016
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np
import scipy.io as io


import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt


from wormdata import shiftData

import wormdata as wd;


## test work data
reload(wd)

wid = 1;
xyd = np.array([[0,0,0,0,1,1,1, 2, 2.1, 2.2, 2.3, 2.1, 2.4, 2.6],[0,0.1,1,1.1,1.2,2, 3,3.1,3.2, 3.1, 3.3, 3.4, 3.1, 3.7]], dtype = float).T;
nn = xyd.shape[0];

w = wd.WormData(xyd, stage = np.ones(nn), valid = np.ones(nn, dtype = bool), label = ('x', 'y'), wid = wid);
w.replaceInvalid();

w.plotTrajectory()

rads = [0.1, 0.5, 1., 2]
npath = 4;
pe = w.calculatePathEntropy(radius = rads, n = npath)

print pe

plt.clf();
nr = pe.shape[1];
for i in range(nr):
  plt.subplot(nr,1,i+1);
  w.plotTrace(shiftData(pe[:,i], s = npath/2));
  plt.title('path entropy radius = %f' % rads[i])


## test on real data

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


reload(wd)

wid = 71;
XYwdata = XYdata[wid].copy();
w = wd.WormData(XYwdata[:,0:2], stage = XYwdata[:,-1], valid = XYwdata[:,0] != 1, label = ('x', 'y'), wid = wid);
w.replaceInvalid();


## average length

lens = w.calculateLengths(n=1)
lens = lens[~np.isnan(lens)];
print lens.mean(), lens.max(), lens.min()

rads = [0.1, 0.5, 1, 2]
npath = 10;
pe = w.calculatePathEntropy(radius = rads, n = npath)

np.save("w%d_path_entropy_n%d.npy" % (wid, npath), pe);


plt.clf();
nr = pe.shape[1];
for i in range(nr):
  plt.subplot(nr,1,i+1);
  w.plotTrace(shiftData(pe[:,i], s = npath/2));
  plt.title('path entropy radius = %f' % rads[i])

def rn(x):
  return x[~np.isnan(x)];

plt.figure(10); plt.clf();
for i in range(nr):
  plt.subplot(nr,1,i+1);
  plt.hist(rn(pe[:,i]))
  plt.title('path entropy radius = %f' % rads[i])


#nr = pe.shape[1];
#for i in range(1): #range(nr):
#  plt.figure(100+i); plt.clf();
#  w.plotDataColor(data = all, c = pe[:,i]);
#  plt.title('path entropy radius = %f' % rads[i])


rots = w.calculateRotations(n=151);

delays = [0, 5, 10, 100, 150];
n_delays = len(delays);
n2_delays = int(np.ceil(n_delays/2.))

for j,r in enumerate(rads):
  fig, axs = plt.subplots(2, n2_delays);
  for i,d in enumerate(delays):
    axs.flat[i].hexbin(pe[:,j], rots[:,d]);
    axs.flat[i].set_title("w=%d d=%d radius=%f" % (wid,d,r));

figname = os.path.join(figdir, 'distances_rotations_wall_s%d.png' % (s))
fig.savefig(figname, bbox_inches = 'tight')


rads = [0.1, 0.5, 1, 2]
npath = 100;
pe2 = w.calculatePathEntropy(radius = rads, n = npath)

np.save("w%d_path_entropy_n%d.npy" % (wid, npath), pe2);


for j,r in enumerate(rads):
  fig, axs = plt.subplots(2, n2_delays);
  for i,d in enumerate(delays):
    axs.flat[i].hexbin(pe2[:,j], rots[:,d]);
    axs.flat[i].set_title("w=%d d=%d radius=%f" % (wid,d,r));

figname = os.path.join(figdir, 'distances_rotations_wall_s%d.png' % (s))
fig.savefig(figname, bbox_inches = 'tight')