# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Aligning the roaming dwelling data set with the image data set and generate roaming data
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np;

import matplotlib.pyplot as plt

import analysis.experiment as exp;
import scripts.preprocessing.file_order as fo;

dir_behaviour = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Code';
dir_roaming   = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/Roaming/Code'

os.chdir(dir_roaming);
import experiment as dexp;
os.chdir(dir_behaviour)

import analysis.plot as fplt

reload(exp)
print 'working at %s' % exp.base_directory

### Load data sets and compare speed

nworms = len(fo.experiment_names);

rd_data = dexp.load_data(strain = 'N2');
rd_speed = rd_data.speed;
rd_speed_th = rd_speed.copy();
th = np.nanpercentile(rd_speed, 95);
rd_speed_th[rd_speed_th > th] = th;

fplt.plot_array(rd_speed_th)


v = [];
for wid in range(nworms):
  v.append(exp.load(strain = 'n2', dtype = 'speed', wid = wid, memmap = None));
ntimes = max([len(vv) for vv in v])


#v = [];
#for wid in range(nworms):
#  print '%d / %d' % (wid, nworms);
#  xy = exp.load(strain = 'N2', dtype = 'xy', wid = wid, memmap = None);
#  xy = xy[np.logical_not(np.any(np.isnan(xy), axis = 1))]  
#  vv = np.linalg.norm(xy[3:,:] - xy[:-3,:], axis = 1);
#  v.append(vv)
  
ntimes = max([len(u) for u in v])
 
im_speed = np.ones((nworms, ntimes)) * np.nan;
for wid, u in enumerate(v):
  #print '%d / %d' % (wid, nworms)
  im_speed[wid, :len(u)] = u;

im_speed_plt = im_speed.copy();
im_speed_plt[np.isnan(im_speed)] = 0.0;
#th = np.percentile(im_speed_plt, 50);
th = 5;
im_speed_plt[im_speed_plt > th] = th;

fplt.plot_array(im_speed_plt)


### Align data sets

import scripts.preprocessing.align_data_sets_util as alg;
reload(alg);

def align(wid = 0, verbose = True):
  
  rv = rd_speed[wid][:rd_data.stage_switch[wid][-1]].copy();
  #iv = im_speed[wid];
  iv = exp.load(strain = 'n2', dtype = 'speed', wid = wid);
  
  print 'worm %d: length: img: %d, rd: %d' % (wid, len(iv), len(rv))
  print 'worm %d: invalid img: %d, rd: %d' % (wid, np.sum(np.isnan(iv)), np.sum(np.isnan(rv)))
    
  # do Shays averaging
  ivm, am = alg.average_invalidation_wrong(iv);
  
  # align data set
  aa = alg.align_indices(ivm, rv, skip = 10, window = 500, search = 50, start = all)
  a = am[aa];
  

  if verbose:
    ivmall = alg.average_invalidation_wrong(iv, invalid = False);  
    fig = plt.figure(10); plt.clf();
    
    ax = plt.subplot(3,1,1);
    plt.plot(ivmall[a]);
    plt.plot(rv)
    plt.title('worm %d, exp %s' % (wid, fo.experiment_names[wid]))
    
    plt.subplot(3,1,2, sharex = ax);
    plt.plot(ivmall[a] - rv)
    
    plt.subplot(3,1,3);
    plt.plot(iv/1.0, 'gray')
    pp = np.ones(len(ivmall)) * np.nan;
    sd = np.sort(np.setdiff1d(range(len(ivmall)), a));
    pp[sd] = ivmall[sd];
    plt.plot(ivmall); plt.plot(pp, 'r')
    sd2 = np.sort(np.setdiff1d(range(len(ivmall)), am));
    pp = np.ones(len(ivmall)) * np.nan;
    pp[sd2] = ivmall[sd2];
    plt.plot(pp, 'k')
    plt.ylim(-1,20)

    plt.pause(0.01);
    
    fig.savefig('align_%d.png' % wid);

  return a;

align(wid = 1, verbose = True)

a = [align(wid= w) for w in range(nworms)]




### worm 71 by hand !!!

wid = 71;
rv = rd_speed[wid][:rd_data.stage_switch[wid][-1]].copy();
iv = exp.load(strain = 'n2', dtype = 'speed', wid = wid);

print 'worm %d: length: img: %d, rd: %d' % (wid, len(iv), len(rv))
print 'worm %d: invalid img: %d, rd: %d' % (wid, np.sum(np.isnan(iv)), np.sum(np.isnan(rv)))

ivm, am = alg.average_invalidation_wrong(iv);

ist = alg.align_start(ivm,rv, verbose = True);
#ist2 = alg.align_start(ivm[ist+45000:],rv[45000:], verbose = True);
#plt.figure(4); plt.clf(); plt.plot(ivm[ist+ist2:]); plt.plot(rv)

reload(alg)
aa = alg.align_indices(ivm, rv, skip = 1, window = 20, search = 3, search_conv = 1200, start = all, max_iter = 5000, 
                       conv_threshold=0.15, conv_precision=None, conv_consistent=5)

fig = plt.figure(10); plt.clf();
ax = plt.subplot(2,1,1);
plt.plot(ivm[ist:], c = 'gray')
plt.plot(rv, c= 'red')
plt.plot(ivm[aa], c = 'blue');
#plt.plot(ivm[ist+369:], c = 'black')
plt.title('worm %d, exp %s' % (wid, fo.experiment_names[wid]))
plt.subplot(2,1,2, sharex = ax);
plt.plot(ivm[aa] - rv)


kst = 44100;
ast = aa[kst]


aa2 = alg.align_indices(ivm[ast:], rv[kst:], skip = 1, window = 200, search = 30, search_conv = 200, start = all, max_iter = 5000, 
                       conv_threshold=0.15, conv_precision=None, conv_consistent=5)

fig = plt.figure(11); plt.clf();
ax = plt.subplot(2,1,1);
plt.plot(ivm[ast:], c = 'gray')
plt.plot(rv[kst:], c= 'red')
plt.plot(ivm[ast:][aa2], c = 'blue');

aanew = aa.copy();
aanew[kst:] = aa2.copy() + ast;
aan = am[aanew];


ivmall = alg.average_invalidation_wrong(iv, invalid = False);  
fig = plt.figure(10); plt.clf();

ax = plt.subplot(3,1,1);
plt.plot(ivmall[aan]);
plt.plot(rv)
plt.title('worm %d, exp %s' % (wid, fo.experiment_names[wid]))

plt.subplot(3,1,2, sharex = ax);
plt.plot(ivmall[aan] - rv)

plt.subplot(3,1,3);
plt.plot(iv/1.0, 'gray')
pp = np.ones(len(ivmall)) * np.nan;
sd = np.sort(np.setdiff1d(range(len(ivmall)), aan));
pp[sd] = ivmall[sd];
plt.plot(ivmall); plt.plot(pp, 'r')
sd2 = np.sort(np.setdiff1d(range(len(ivmall)), am));
pp = np.ones(len(ivmall)) * np.nan;
pp[sd2] = ivmall[sd2];
plt.plot(pp, 'k')
plt.ylim(-1,20)

plt.pause(0.01);

fig.savefig('align_%d.png' % wid);


a[wid] = aan;


### Save mapping to file

fn = os.path.join(exp.data_directory, 'n2_roam_align_w=%d_s=all.npy');



for wid in range(nworms):
  print '%d / %d' % (wid, nworms)
  np.save(fn % wid, a[wid])

fn = exp.filename(strain = 'n2', wid = 0, dtype = 'roam_align')

### Generate Roaming Data

for wid in range(nworms):
  print '%d / %d' % (wid, nworms)
  roam = np.array(rd_data.roam[wid][:rd_data.stage_switch[wid][-1]].copy(), dtype = bool);
  fn = exp.filename(strain = 'n2', wid = wid, dtype = 'roam');
  roamall = np.ones(len(v[wid])) * np.nan;
  roamall[a[wid]] = roam;
  np.save(fn, roamall)



