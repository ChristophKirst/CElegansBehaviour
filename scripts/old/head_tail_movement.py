# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:14:43 2016

@author: ckirst
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import analysis.experiment as exp
import analysis.plot as aplt
import worm.model as wmodel


npts = 22;
w = wmodel.WormModel(npoints = npts);

fn = exp.filename(strain = 'n2', wid = 80, dtype = 'img')
imgdata = np.load(fn, mmap_mode = 'r');

ntimes = imgdata.shape[0];

theta, orientation, xy, length = w.theta();
ntheta = theta.shape[0];

data_path = '/home/ckirst/Desktop';
data_names = ['theta', 'orientation', 'xy', 'length', 'center', 'width'];
data_sizes = [(ntimes, ntheta), ntimes, (ntimes,2), ntimes, (ntimes, npts, 2), (ntimes, npts)];

data_files = [os.path.join(data_path, 'analysis_2016_10_26_' + n + '.npy') for n in data_names];

fig_dir = '/home/ckirst/Desktop/labmeeting'


### match head and tails


data = {n : np.load(f, mmap_mode = 'r+') for n,f in zip(data_names, data_files)};


theta = data["theta"];
center = data["center"]
orient = data["orientation"]
pos = exp.load(wid = 80, dtype = 'xy')

#head vs center pos
c_pos = pos + data["xy"];
h_pos = pos + data["center"][:,0];
t_pos = pos + data["center"][:,-1];
valid = data["width"][:,0] != -1;


plt.figure(1); plt.clf();
t0 = 0; t1 = 1000;
plt.plot(t_pos[t0:t1,0], t_pos[t0:t1,1])
plt.plot(h_pos[t0:t1,0], h_pos[t0:t1,1])

#match head tail postions

match = np.zeros(ntimes, dtype = int);
for i in range(1, ntimes):
  if i%1000==0:
    print '%d/%d' % (i, ntimes)
  if valid[i-1] and valid[i]:
    h0 = h_pos[i-1]; h1 = h_pos[i];
    t0 = t_pos[i-1]; t1 = t_pos[i];
    dists_h0 = [np.linalg.norm(h0-h1), np.linalg.norm(h0-t1)];
    dists_t0 = [np.linalg.norm(t0-h1), np.linalg.norm(t0-t1)];
    h0_match = np.argmin(dists_h0);
    t0_match = np.argmin(dists_t0);
    
    if h0_match == t0_match: #same minimal distance to next point, maximize max distance
      if h0_match == 0:
        k = np.argmin([dists_h0[1], dists_t0[1]]);
        if k == 0: #h0->t1, t0->h1
          match[i] = 1;
        else: #t0 -> t1
          match[i] = 0;
      else: # h0_match == 1:
        k = np.argmin([dists_h0[0], dists_t0[0]]);
        if k == 0: #h0->h1
          match[i] = 0;
        else: #t0 -> h1
          match[i] = 1;
    else:
      if h0_match == 0: #t0_match==1
        match[i] = 0;
      else: #h0_mathc ==1, t0_match == 0
        match[i] = 1; # reverse
  else:
    match[i] = -1;


# align data from matches

h_pos_a = h_pos.copy();
t_pos_a = t_pos.copy();


reverse = np.zeros(ntimes);

sw = False;
for i in range(ntimes):
  if match[i] == 1: #switch
    sw = not(sw);
  #if match[i] == -1: # interrupted -> keep switching as before -> does not matter
  if sw: # switch
    htmp  = h_pos_a[i].copy();
    h_pos_a[i] = t_pos_a[i];
    t_pos_a[i] = htmp;
  
  reverse[i] = sw;
    

# calculate variance and assign head to max variance



idx = match == -1;
swi = np.where(idx)[0];
swi_s = swi[:-1];
swi_e = swi[1:];


def calc_change(pos):
  if len(pos) <= 2:
    return 0;
  else:
    d = np.linalg.norm(pos[2:]-pos[:-2], axis = 1);
    return np.mean(d);
    
for i in range(len(swi_s)):
  ch = calc_change(h_pos_a[swi_s[i]:swi_e[i]]);
  ct = calc_change(t_pos_a[swi_s[i]:swi_e[i]]);
  
  if ct > ch: #switch head tail 
    h_pos_a[swi_s[i]:swi_e[i]], t_pos_a[swi_s[i]:swi_e[i]] = t_pos_a[swi_s[i]:swi_e[i]].copy(), h_pos_a[swi_s[i]:swi_e[i]].copy();
    reverse[swi_s[i]:swi_e[i]] = 1 - reverse[swi_s[i]:swi_e[i]];
    
  

fig = plt.figure(2); plt.clf();
t0 =100000; t1 = 103000;
ii = np.arange(t0,t1)[valid[t0:t1]]
plt.plot(t_pos_a[ii,0], t_pos_a[ii,1], 'gray')
plt.plot(h_pos_a[ii,0], h_pos_a[ii,1], 'k')

fig.savefig(os.path.join(fig_dir, 'sample_feeding.pdf'), facecolor = 'white')

fig = plt.figure(2); plt.clf();
t0 =303000; t1 = t0+5000;
ii = np.arange(t0,t1)[valid[t0:t1]]
plt.plot(t_pos_a[ii,0], t_pos_a[ii,1], 'gray', linewidth = 1)
plt.plot(h_pos_a[ii,0], h_pos_a[ii,1], 'k', linewidth = 1)

plt.scatter(h_pos_a[ii,0], h_pos_a[ii,1], c = ii, cmap = 'spectral', edgecolor= 'face', s = 10)
plt.axis('equal')

fig.savefig(os.path.join(fig_dir, 'title.png'), facecolor = 'white')



i = 0;

for k in range(20):
  fig = plt.figure(2); plt.clf();
  t0 =403000 + i; t1 = t0+5000; i+=5000
  ii = np.arange(t0,t1)[valid[t0:t1]]
  plt.plot(t_pos_a[ii,0], t_pos_a[ii,1], 'gray', linewidth = 1)
  plt.plot(h_pos_a[ii,0], h_pos_a[ii,1], 'k', linewidth = 1)
  
  plt.scatter(h_pos_a[ii,0], h_pos_a[ii,1], c = ii, cmap = 'spectral', edgecolor= 'face', s = 10)
  plt.axis('equal')
  
  fig.savefig(os.path.join(fig_dir, 'head_tail_sample_traces_%d.png' % k), facecolor = 'white')





dd = np.linalg.norm(h_pos_a[ii]-t_pos_a[ii], axis = 1)

plt.figure(70); plt.clf();
plt.plot(dd)


di = 100;
fig = plt.figure(2); plt.clf();
t0 =308000 + di; t1 = t0+300;
di+=300;
ii = np.arange(t0,t1)[valid[t0:t1]]
plt.plot(t_pos_a[ii,0], t_pos_a[ii,1], 'gray', linewidth = 1)
plt.plot(h_pos_a[ii,0], h_pos_a[ii,1], 'k', linewidth = 1)
plt.scatter(h_pos_a[ii,0], h_pos_a[ii,1], c = ii, cmap = 'spectral', edgecolor= 'face', s = 10)
plt.axis('equal')


dd = np.linalg.norm(h_pos_a[ii]-t_pos_a[ii], axis = 1)

plt.figure(70); plt.clf();
plt.plot(dd)



#show the movie

imgs = exp.load_img(t=range(t0,t1));
#imgs = exp.load_img(t=ii);

import pyqtgraph as pg

pg.show(np.transpose(imgs, [0,1,2]))






### pca on oriented head / tail data


center = data["center"].copy();

for i in range(ntimes):
  if reverse[i]:
    center[i] = center[i,::-1];

center_file = os.path.join(data_path, 'oriented_center.npy')
np.save(center_file, center)

#calculate new thetas etc

theta_file = os.path.join(data_path, 'oriented_theta.npy')
orien_file = os.path.join(data_path, 'oriented_orientation.npy')
xy_file = os.path.join(data_path, 'oriented_xy.npy')
theta = np.zeros((ntimes, npts-2));
np.save(theta_file, theta)
orien = np.zeros((ntimes));
np.save(orien_file, orien)
xx = np.zeros((ntimes, 2));
np.save(xy_file, xx)

def calc_theta(i):
  if i%10000==0: 
    print 'processing %d/%d' % (i,ntimes)
  th = np.load(theta_file, mmap_mode = 'r+');
  oo = np.load(orien_file, mmap_mode = 'r+');
  xx = np.load(xy_file, mmap_mode = 'r+');
  c = np.load(center_file, mmap_mode = 'r');
  
  theta, orientation, xy, length = wgeo.theta_from_center_discrete(c[i]);
  th[i] = theta;
  oo[i] = orientation;
  xx[i] = xy;
  

from multiprocessing import Pool, cpu_count;

pool = Pool(processes = cpu_count());

pool.map(calc_theta, range(ntimes));


  



### PCA

theta = np.load(theta_file);

#plt.figure(12); plt.clf();
#aplt.plot_pca(theta)

pca = aplt.PCA(theta)

pca_comp = pca.Wt;

import worm.model as wm
import worm.geometry as wgeo
w = wm.WormModel(npoints = 22)

fig = plt.figure(170); plt.clf();
for i in range(len(pca_comp)):
  tt = pca_comp[i] * 21;
  cc = wgeo.center_from_theta_discrete(tt, length = 95);
  dd = np.sum(np.diff(cc, axis = 0), axis = 0);
  oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;

  w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
  plt.subplot(5,4,i+1)
  plt.title('PCA %d' % (i+1))
  w.plot(ccolor='b')
  plt.axis('equal')

fig.savefig(os.path.join(fig_dir, 'theta_pca_oriented.png'), facecolor='white')
   
fig = plt.figure(170); plt.clf();
for i in range(9):
  tt = pca_comp[i] * 21;
  cc = wgeo.center_from_theta_discrete(tt, length = 95);
  dd = np.sum(np.diff(cc, axis = 0), axis = 0);
  oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;

  w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
  plt.subplot(3,3,i+1)
  plt.title('PCA %d' % (i+1))
  w.plot(ccolor='b')
  plt.axis('equal')
  
  
fig.savefig(os.path.join(fig_dir, 'theta_pca_9_oriented.png'), facecolor='white')
   
   
   
     
fig = plt.figure(180); plt.clf();
for i in range(len(pca_comp)):
  tt = pca_comp[i];
  #tta = tt - np.mean(tt);
  plt.subplot(5,4,i+1)
  plt.plot(tt, linewidth = 2, c = 'r')
  plt.title('PCA %d' % (i+1))
  plt.ylim(-0.6, 0.6)

fig.savefig(os.path.join(fig_dir, 'theta_pca_plot.png'),  facecolor='white')
       

fig = plt.figure(180); plt.clf();
for i in range(9):
  tt = pca_comp[i];
  #tta = tt - np.mean(tt);
  plt.subplot(3,3,i+1)
  plt.plot(tt, linewidth = 2, c = 'r')
  plt.title('PCA %d' % (i+1))
  plt.ylim(-0.6, 0.6)

fig.savefig(os.path.join(fig_dir, 'theta_pca_9_plot.png'),  facecolor='white')
      



pcs = pca.Y;
pcs.shape


aplt.plot_array(pcs.T)

fig = plt.figure(190); plt.clf();
#plt.imshow(pca.Wt, interpolation = 'none', aspect = 'auto', cmap = 'viridis')
#plt.colorbar(pad = 0.01,fraction = 0.01)
#plt.title('pca vectors');
plt.plot(np.cumsum(pca.fracs), 'k', linewidth = 1)  
plt.ylim(0,1)
  
fig.savefig(os.path.join(fig_dir, 'theta_pca_variance.png'),  facecolor='white')  
  
  

## time evolution of components

fig = plt.figure(200); plt.clf();
for i in range(6):
  tt = pca_comp[i];
  #tta = tt - np.mean(tt);
  plt.subplot(3,2,i+1)
  plt.plot(pcs[:,i], linewidth = 2, c = 'k')
  plt.title('PCA %d' % (i+1))
  plt.ylim(-10, 10)
  plt.xlim(0, len(pcs))  
# some basic statistics / distributions
fig.savefig(os.path.join(fig_dir, 'theta_pca_time.png'),  facecolor='white')   




#example trace + pcs
di = 100;
fig = plt.figure(270); plt.clf();
ax = plt.subplot2grid((4,8),(0,0),rowspan = 4, colspan = 4);
t0 =308000 + di; t1 = t0+1500;
di+=1500;
ii = np.arange(t0,t1)[valid[t0:t1]]
ax.plot(t_pos_a[ii,0], t_pos_a[ii,1], 'gray', linewidth = 1)
ax.plot(h_pos_a[ii,0], h_pos_a[ii,1], 'k', linewidth = 1)
ax.scatter(h_pos_a[ii,0], h_pos_a[ii,1], c = ii, cmap = 'spectral', edgecolor= 'face', s = 10)
plt.axis('equal')

for i in range(4):
  ax = plt.subplot2grid((4,8),(i,4), colspan = 3);
  ax = plt.plot(pcs[ii,i], linewidth = 2, c = 'k')
  
  ax = plt.subplot2grid((4,8),(i,7))
  w = wm.WormModel(npoints = 22)
  tt = pca_comp[i] * 21;
  cc = wgeo.center_from_theta_discrete(tt, length = 95);
  dd = np.sum(np.diff(cc, axis = 0), axis = 0);
  oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;

  w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
  w.plot(ccolor='b')
  plt.axis('equal')


fig.savefig(os.path.join(fig_dir, 'pca_dynamics_roaming_diffusive.png'),  facecolor='white')   



# examples young

di = 100;
fig = plt.figure(270); plt.clf();
ax = plt.subplot2grid((4,8),(0,0),rowspan = 4, colspan = 4);
t0 =8000 + di; t1 = t0+1500;
di+=1500;
ii = np.arange(t0,t1)[valid[t0:t1]]
ax.plot(t_pos_a[ii,0], t_pos_a[ii,1], 'gray', linewidth = 1)
ax.plot(h_pos_a[ii,0], h_pos_a[ii,1], 'k', linewidth = 1)
ax.scatter(h_pos_a[ii,0], h_pos_a[ii,1], c = ii, cmap = 'spectral', edgecolor= 'face', s = 10)
plt.axis('equal')

for i in range(4):
  ax = plt.subplot2grid((4,8),(i,4), colspan = 3);
  ax = plt.plot(pcs[ii,i], linewidth = 2, c = 'k')
  
  ax = plt.subplot2grid((4,8),(i,7))
  w = wm.WormModel(npoints = 22)
  tt = pca_comp[i] * 21;
  cc = wgeo.center_from_theta_discrete(tt, length = 95);
  dd = np.sum(np.diff(cc, axis = 0), axis = 0);
  oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;

  w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
  w.plot(ccolor='b')
  plt.axis('equal')


fig.savefig(os.path.join(fig_dir, 'pca_dynamics_roaming_long_diffusive.png'),  facecolor='white')   





### stage resolved PCA


theta = np.load(theta_file);

#theta = np.load(data_files[0]);
#theta *= 1.0/(theta.shape[1]+1)

w80s = np.load(exp.filename(strain = 'n2', dtype = 'stage', wid = 80));

valid = data["width"][:,0] != -1;

for s in range(1,6):
  idx = np.logical_and(valid, w80s == s);
    
  pca = aplt.PCA(theta[idx]);
  pca_comp = pca.Wt;

  w = wm.WormModel(npoints = 22)

  plt.figure(20+s); plt.clf();
  for i in range(len(pca_comp)):
    plt.subplot(5,4,i+1)
    w = wm.WormModel(npoints = 22)
    tt = pca_comp[i] * 21;
    cc = wgeo.center_from_theta_discrete(tt, length = 95);
    dd = np.sum(np.diff(cc, axis = 0), axis = 0);
    oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;
  
    w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
    w.plot(ccolor='b')
    plt.axis('equal')
    plt.title('PCA %d' % (i+1))


  fig.savefig(os.path.join(fig_dir, 'theta_pca_stage=%d.png' % s),  facecolor='white'); 



fig2 = plt.figure(190); plt.clf();
for s in range(1,6):
  idx = np.logical_and(valid, w80s == s);
    
  pca = aplt.PCA(theta[idx]);
  pca_comp = pca.Wt;

  w = wm.WormModel(npoints = 22)

  fig = plt.figure(20+s); plt.clf();
  for i in range(9):
    plt.subplot(3,3,i+1)
    w = wm.WormModel(npoints = 22)
    tt = pca_comp[i] * 21;
    cc = wgeo.center_from_theta_discrete(tt, length = 95);
    dd = np.sum(np.diff(cc, axis = 0), axis = 0);
    oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;
  
    w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
    w.plot(ccolor='b')
    plt.axis('equal')
    plt.title('PCA %d' % (i+1))


  fig.savefig(os.path.join(fig_dir, 'theta_pca_stage=%d.png' % s),  facecolor='white'); 

  
  fig2 = plt.figure(190);
  #plt.imshow(pca.Wt, interpolation = 'none', aspect = 'auto', cmap = 'viridis')
  #plt.colorbar(pad = 0.01,fraction = 0.01)
  #plt.title('pca vectors');
  plt.plot(np.cumsum(pca.fracs), linewidth = 1)  
  plt.ylim(0,1)

fig2.savefig(os.path.join(fig_dir, 'theta_pca_variance_stages.png'),  facecolor='white')  



fig2 = plt.figure(190); plt.clf();
for s in range(1,6):
  idx = np.logical_and(valid, w80s == s);
    
  pca = aplt.PCA(theta[idx]);
  pca_comp = pca.Wt;

  w = wm.WormModel(npoints = 22)

  fig = plt.figure(20+s); plt.clf();
  for i in range(9):
    plt.subplot(3,3,i+1)
    w = wm.WormModel(npoints = 22)
    tt = pca_comp[i] * 21;
    cc = wgeo.center_from_theta_discrete(tt, length = 95);
    dd = np.sum(np.diff(cc, axis = 0), axis = 0);
    oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;
  
    w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
    w.plot(ccolor='b')
    plt.axis('equal')
    plt.title('PCA %d' % (i+1))


  fig.savefig(os.path.join(fig_dir, 'theta_pca_stage=%d.png' % s),  facecolor='white'); 

  
  fig2 = plt.figure(190);
  #plt.imshow(pca.Wt, interpolation = 'none', aspect = 'auto', cmap = 'viridis')
  #plt.colorbar(pad = 0.01,fraction = 0.01)
  #plt.title('pca vectors');
  plt.plot(np.cumsum(pca.fracs), linewidth = 1)  
  plt.ylim(0,1)

fig2.savefig(os.path.join(fig_dir, 'theta_pca_variance_stages.png'),  facecolor='white')  





for s in range(1,6):
  idx = np.logical_and(valid, w80s == s);
    
  pca = aplt.PCA(theta[idx]);
  pca_comp = pca.Wt;

  w = wm.WormModel(npoints = 22)

  fig = plt.figure(40+s); plt.clf();
  for i in range(9):
    plt.subplot(3,3,i+1)
    w = wm.WormModel(npoints = 22)
    tt = pca_comp[i] * 21;
    cc = wgeo.center_from_theta_discrete(tt, length = 95);
    dd = np.sum(np.diff(cc, axis = 0), axis = 0);
    oo = np.arctan2(dd[1], dd[0]) + 0* np.pi;
  
    w.center = wgeo.center_from_theta_discrete(tt, length = 95, orientation = -oo);
    #w.plot(ccolor='b')
    plt.plot(tt, linewidth = 1, c = 'r')
    #plt.axis('equal')
    plt.xlim(0,19);
    plt.ylim(-12,12)
    plt.title('PCA %d' % (i+1))


  fig.savefig(os.path.join(fig_dir, 'theta_pca_stage=%d_plot.png' % s),  facecolor='white'); 







### PCA components statistics

#todo






### width growth


length = data['length'];

dl = np.diff(length)
dl = np.hstack([[0], dl])

ids = np.abs(dl) < 5;
ids = np.logical_and(length > 15, ids)

fig = plt.figure(300); plt.clf();
#plt.plot(length[ids])


window = 200;
kernel = np.repeat(1.0, window)/window
length_smth = np.convolve(length[ids], kernel, 'same')

plt.plot(length_smth[:-150])
plt.xlim(0, len(length_smth[:-150]))

fig.savefig(os.path.join(fig_dir, 'length.png'),  facecolor='white')   



width = data['width'];
width = width[ids]

#average width profile
fig = plt.figure(301); plt.clf();
#plt.plot(np.mean(width, axis = 0), 'r')
#plt.ylim(0,10)

bds = np.array(np.linspace(0, len(width), 75))
ww = np.zeros((len(bds)-1, width.shape[1]))
for i in range(len(bds)-1):
  ww[i] = np.mean(width[bds[i]:bds[i+1]], axis = 0)
plt.imshow(ww.T, cmap = 'Reds', aspect = 'auto')
#plt.plot(np.mean(width[-100000:], axis = 0))

fig.savefig(os.path.join(fig_dir, 'width.png'),  facecolor='white')   


fig = plt.figure(302); plt.clf();
plt.plot(np.mean(width, axis = 0), 'r')
plt.ylim(0,7); plt.xlim(0,21)

fig.savefig(os.path.join(fig_dir, 'width_profile.png'),  facecolor='white')   







### some basic statistics for head tail movement

#how much does the head move more w.r.t tail



length = data['length'];
vld = data['width'][:,0]!=-1;
vld = np.logical_and(vld, length > 15)
nvld = np.logical_not(vld)

hp = h_pos_a; hp[nvld] = np.nan;
tp = t_pos_a; tp[nvld] = np.nan;

dh = hp[10:] - hp[:-10];
dt = tp[10:] - tp[:-10];

idx = np.any(np.logical_not(np.isnan(dh)), axis = 1)

dh2 = np.linalg.norm(dh, axis = 1);
dt2 = np.linalg.norm(dt, axis = 1);

idxh = np.logical_and(np.any(np.logical_not(np.isnan(dh)), axis = 1), dh2 < 40);
idxt = np.logical_and(np.any(np.logical_not(np.isnan(dh)), axis = 1), dt2 < 40);

plt.figure(1); plt.clf()
plt.plot(dh2);
plt.plot(dt2);

fig = plt.figure(2); plt.clf()
plt.hist([dh2[idxh],dt2[idxt]] , bins = 256, stacked = False, histtype = 'step', color = ['red', 'gray'], normed = True);
plt.legend(['head', 'tail'])
plt.xlim(0,20)

fig.savefig(os.path.join(fig_dir, 'head_tail_statistics.png'),  facecolor='white'); 


#stage resolved


fig = plt.figure(78); plt.clf();
for s in range(1,6):
  plt.subplot(2,3,s);
  
  idxh = np.logical_and(np.any(np.logical_not(np.isnan(dh)), axis = 1), dh2 < 40);
  idxt = np.logical_and(np.any(np.logical_not(np.isnan(dh)), axis = 1), dt2 < 40);

  idxh = np.logical_and(idxh, w80s[:len(idxh)] == s )
  idxt = np.logical_and(idxt, w80s[:len(idxh)] == s )

  plt.hist([dh2[idxh],dt2[idxt]] , bins = 256, stacked = False, histtype = 'step', color = ['red', 'gray'], normed = True);
  plt.xlim(0,20)
  plt.title('S%d'%s)

idxh = np.logical_and(np.any(np.logical_not(np.isnan(dh)), axis = 1), dh2 < 40);
idxt = np.logical_and(np.any(np.logical_not(np.isnan(dh)), axis = 1), dt2 < 40);

plt.subplot(2,3,6);
plt.hist([dh2[idxh],dt2[idxt]] , bins = 256, stacked = False, histtype = 'step', color = ['red', 'gray'], normed = True);
plt.legend(['head', 'tail'])
plt.title('all')
plt.xlim(0,20)


fig.savefig(os.path.join(fig_dir, 'head_tail_statistics.png'),  facecolor='white'); 

 
cov = np.cov(theta.T) 
 
fig = plt.figure(56); plt.clf();
plt.imshow(cov,origin = 'lower', cmap = 'jet')
plt.colorbar(pad = 0.05, fraction = 0.05)


fig.savefig(os.path.join(fig_dir, 'theta_cov.png'),  facecolor='white'); 


 
 










fig.savefig(os.path.join(fig_dir, 'sample_feeding.pdf'), facecolor = 'white')



plt.figure(2); plt.clf();
t0 =200000; t1 = t0 + 1000;
plt.subplot(1,2,1);
plt.plot(t_pos_a[t0:t1,0]-pos[t0:t1,0], t_pos_a[t0:t1,1]-pos[t0:t1,1])
plt.subplot(1,2,2)
plt.plot(h_pos_a[t0:t1,0]-pos[t0:t1,0], h_pos_a[t0:t1,1]-pos[t0:t1,1])



plt.figure(2); plt.clf();
t0 =200000; t1 = t0 + 1000;
plt.subplot(1,2,1);
plt.plot(t_pos[t0:t1,0]-pos[t0:t1,0], t_pos[t0:t1,1]-pos[t0:t1,1])
plt.subplot(1,2,2)
plt.plot(h_pos[t0:t1,0]-pos[t0:t1,0], h_pos[t0:t1,1]-pos[t0:t1,1])



dd = np.linalg.norm(h_pos_a[t0:t1]-t_pos_a[t0:t1], axis = 1)


plt.figure(70); plt.clf();
plt.plot(dd)


i = np.where(dd < 5)[0];

img = exp.load_img(t = t0 + i);
plt.figure(71); plt.clf(); plt.imshow(img[0])

plr.figure(100); plt.clf();
w.from_image(img[0], verbose = True)

import worm.geometry as wgeo
reload(wgeo)

plt.figure(71); plt.clf();
wgeo.shape_from_image(img[3], verbose = True);

  

from multiprocessing import Pool, cpu_count;

pool = Pool(processes = cpu_count()-2);

pool.map(process, range(ntimes));



data = [np.load(os.path.join(data_path, 'analysis_2016_10_25_' + n + '.npy'), mmap_mode = 'r+') for n in data_names];

### Failed frames
nfailed = np.sum(data[3] == -1);
print 'percentage failed %f' % (100.0 * nfailed/data[3].shape[0])

