\# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Matlab to Numpy Data Conversion Routines for Worm Data
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import glob
import matplotlib.pyplot as plt;
import scipy.io
import numpy as np

import analysis.experiment as exp

### Convert Coordinates in Folder

base_directory = '/run/media/ckirst/My Book'

fig_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Figures/2017_01_20_TimeAnalysis/'

worm_directories = sorted(glob.glob(os.path.join(base_directory, 'Results*/*/')))


### Time stamps of the data

import datetime, time, calendar

exp_times = np.zeros(len(worm_directories));

for wid, wdir in enumerate(worm_directories):
  file_pattern = os.path.join(wdir, 'corrd*.mat');
  fns = sorted(glob.glob(file_pattern));
  fn = fns[0]; 
  date = fn[-26:-9];
  t = time.strptime(date, '%Y-%m-%d-%H%M%S');
  secs = calendar.timegm(t);
  
  secs_24h = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec;
  
  exp_times[wid] = secs_24h;  


fig = plt.figure(1); plt.clf();
plt.plot(exp_times)
plt.xlim(0, len(exp_times));
plt.ylabel('day time [sec]')

fig.savefig(os.path.join(fig_directory, 'experiments_starting_times_24hr.pdf'))


fn_times = os.path.join(exp.experiment_directory, 'time_stamps.npy');
np.save(fn_times, exp_times);


### Activity vs time stamps

# settings
fn_times = os.path.join(exp.experiment_directory, 'time_stamps.npy');
exp_times = np.load(fn_times)

nworms = len(exp_times);

rate = 3; # Hz sample rate

time_bin = 1*30; # s;

nbins = rate * time_bin; # number of time bins to calculate mean
ntimes = 5 * 24 * 60 * 60 / time_bin;  # totla number of time steps in analysis
v_24hr = np.zeros((nworms, ntimes)) * np.nan;
r_24hr = np.zeros((nworms, ntimes)) * np.nan;


# calculate data
for wid in range(nworms):
  xy = exp.load(strain = 'N2', dtype = 'xy', wid = wid, memmap=None);
  rt = exp.load(strain = 'N2', dtype = 'phi',wid = wid, memmap=None);
  
  # invalid positions
  inv = xy[:,0] == 1.0;
  xy[inv,:] = np.nan;
  
  # calculate speed
  dxy = xy[1:,:] - xy[:-1,:];
  v = np.sqrt(dxy[:,0]**2 + dxy[:,1]**2) * rate;
  nv = len(v);
  
  # bin speed
  nvm = nv/nbins * nbins;
  v_mean = v[:nvm];
  v_mean = v_mean.reshape([nvm/nbins, nbins]);
  v_mean = np.nanmean(v_mean, axis = 1);
  v_mean = np.hstack([v_mean, [np.nanmean(v[nvm:])]]);
  
  # bin rot
  r_mean = rt[:nvm];
  r_mean = r_mean.reshape([nvm/nbins, nbins]);
  r_mean = np.nanmean(r_mean, axis = 1);
  r_mean = np.hstack([r_mean, [np.nanmean(rt[nvm:])]]);
  
  # time bin
  ts = int(exp_times[wid] / time_bin);
  v_24hr[wid, ts:ts+len(v_mean)] = v_mean.copy();
  r_24hr[wid, ts:ts+len(r_mean)] = r_mean.copy();


# figure speed
fig = plt.figure(2); plt.clf();
vrange = np.nanpercentile(v_24hr, [5, 95]);
plt.imshow(v_24hr, aspect = 'auto', vmax = vrange[1], interpolation = 'none', cmap = plt.cm.viridis)

nhr  = 60 * 60 / time_bin;
nday = 24 * nhr;
for d in range(ntimes/nhr):
  plt.plot([d*nhr, d*nhr], [-0.5, len(v_24hr)-0.5], 'k')
for d in range(ntimes/nday):
  plt.plot([d*nday, d*nday], [-0.5, len(v_24hr)-0.5], 'k', linewidth = 2.5)

plt.xlim(0, v_24hr.shape[1]); 
plt.ylim(-0.5, v_24hr.shape[0]-0.5);

days = np.linspace(0, v_24hr.shape[1] , 6);
labl = ['%d'%d for d  in 24 *np.linspace(0,5, 6)];
plt.xticks(days, labl);
plt.xlabel('absolute day time [hrs]'); 
plt.ylabel('worm id');

cb = plt.colorbar(fraction = 0.025, shrink = 0.5, pad = 0.01)
cb.ax.set_ylabel('speed [a.u.]', rotation=270, labelpad = 20)
plt.tight_layout()

fig.savefig(os.path.join(fig_directory, 'time_analysis_24hr_speed.pdf'))


# figure rotation
fig = plt.figure(2); plt.clf();
vrange = np.nanpercentile(r_24hr, [5, 95]);
plt.imshow(r_24hr, aspect = 'auto', vmax = vrange[1], interpolation = 'none', cmap = plt.cm.viridis)

# hr lines
for d in range(ntimes/nhr):
  plt.plot([d*nhr, d*nhr], [-0.5, len(r_24hr)-0.5], 'k')
for d in range(ntimes/nday):
  plt.plot([d*nday, d*nday], [-0.5, len(r_24hr)-0.5], 'k', linewidth = 2.5)

plt.xlim(0, r_24hr.shape[1]);
plt.ylim(-0.5, r_24hr.shape[0]-0.5);

days = np.linspace(0, r_24hr.shape[1] , 6);
labl = ['%d'%d for d  in 24 *np.linspace(0,5, 6)];
plt.xticks(days, labl)
plt.xlabel('absolute day time [hrs]')
plt.ylabel('worm id')

cb = plt.colorbar(fraction = 0.025, shrink = 0.5, pad = 0.01)
cb.ax.set_ylabel('speed [a.u.]', rotation=270, labelpad = 20)
plt.tight_layout()

fig.savefig(os.path.join(fig_directory, 'time_analysis_24hr_rotation.pdf'))




### Plot trajectory around area of interest

wid = 42;

istr = 1185;
iend = 1500;

ts = int(exp_times[wid] / time_bin);

jstr = (istr - ts) * time_bin * rate;
jend = (iend - ts) * time_bin * rate;

xy = exp.load(strain = 'N2', wid = wid, memmap = None);
inv = xy[:,0] <= 1.0; xy[inv,:] = np.nan;

plt.figure(5); plt.clf();
plt.scatter(xy[jstr:jend, 0], xy[jstr:jend, 1])






### Roaming Fraction

rate = 3; # Hz sample rate

time_bin = 15*60; # s;
time_bin = 60 * 5; # s;

nbins = rate * time_bin; # number of time bins to calculate mean
ntimes = 5 * 24 * 60 * 60 / time_bin;  # totla number of time steps in analysis
roam_24hr = np.zeros((nworms, ntimes));
r_24hr = np.zeros((nworms, ntimes));
v_24hr = np.zeros((nworms, ntimes));

for wid in range(nworms):  
  # bin speed
  roam = exp.load(strain = 'N2', dtype = 'roam',wid = wid, memmap=None);
  nv = len(roam);
  nvm = nv/nbins * nbins;
  roam_mean = roam[:nvm];
  roam_mean = roam_mean.reshape([nvm/nbins, nbins]);
  roam_mean = np.nanmean(roam_mean, axis = 1);
  roam_mean = np.hstack([roam_mean, [np.nanmean(roam[nvm:])]]);  
    
  ts = int(exp_times[wid] / time_bin);
  roam_24hr[wid, ts:ts+len(roam_mean)] = roam_mean.copy();
  
  
  # bin rotation
  r = exp.load(strain = 'N2', dtype = 'phi',wid = wid, memmap=None);
  nv = len(r);
  nvm = nv/nbins * nbins;
  
  r_mean = r[:nvm];
  r_mean = r_mean.reshape([nvm/nbins, nbins]);
  r_mean = np.nanmean(r_mean, axis = 1);
  r_mean = np.hstack([r_mean, [np.nanmean(r[nvm:])]]);
  
  ts = int(exp_times[wid] / time_bin);
  r_24hr[wid, ts:ts+len(r_mean)] = r_mean.copy();
  
  
  # bin velocity
  v = exp.load(strain = 'N2', dtype = 'xy',wid = wid, memmap=None);
  v = np.linalg.norm(v[1:, :] - v[:1, :], axis = 1);
  nv = len(v);
  nvm = nv/nbins * nbins;
  
  v_mean = v[:nvm];
  v_mean = v_mean.reshape([nvm/nbins, nbins]);
  v_mean = np.nanmean(v_mean, axis = 1);
  v_mean = np.hstack([v_mean, [np.nanmean(v[nvm:])]]);
  
  ts = int(exp_times[wid] / time_bin);
  v_24hr[wid, ts:ts+len(r_mean)] = v_mean.copy();  
  

plt.figure(12); plt.clf();
#plt.plot(v_mean)
#plt.ylim(0,20)
np.nanmax(roam_24hr)
rrange = np.nanpercentile(roam_24hr, [5, 95]);
plt.imshow(roam_24hr, aspect = 'auto', vmax = rrange[1], interpolation = 'none')
plt.tight_layout()

# hr lines
nhr  = 60 * 60 / time_bin;
nday = 24 * nhr;

for d in range(ntimes/nhr):
  plt.plot([d*nhr, d*nhr], [-0.5, len(roam_24hr)-0.5], 'k')

for d in range(ntimes/nday):
  plt.plot([d*nday, d*nday], [-0.5, len(roam_24hr)-0.5], 'r', linewidth = 2)

plt.xlim(0, roam_24hr.shape[1]);
plt.ylim(-0.5, roam_24hr.shape[0]-0.5);




### Align romaing to different transitions

phase_names = ['hatch', 'L1', 'L2', 'L3', 'L4', 'L5'];

for tr in range(5):
  fig = plt.figure(400+tr); plt.clf();
  roam_24r_algn = np.zeros((roam_24hr.shape[0], roam_24hr.shape[1]+0));
  tref = np.max(transitions_cor[:, tr]);
  for wid in range(nworms):
    tt = transitions_cor[wid, tr];
    ts =  transitions_cor[wid, 0];
    te =  transitions_cor[wid, -1];      
    nt = te - ts;
    roam_24r_algn[wid, ts + tref - tt : ts + tref-tt + nt] = roam_24hr[wid,  ts:te];
  
  #rmax = np.nanpercentile(roam_24hr, 95); 
  rmax = 1;
  plt.imshow(roam_24r_algn, interpolation = 'none', aspect = 'auto', cmap = plt.cm.viridis, vmax = rmax )
  cb = plt.colorbar(fraction = 0.025, shrink = 0.5, pad = 0.01)
  cb.ax.set_ylabel('rotation [au]', rotation=270, labelpad = 20)
  
  days = np.linspace(0, roam_24hr.shape[1] , 6);
  labl = ['%d'%d for d  in 24 *np.linspace(0,5, 6)];  
  plt.xticks(days, labl);
  plt.xlabel('time [hrs]');   
  plt.ylabel('worm id');
  plt.tight_layout()
  
  plt.title(phase_names[tr]);
  plt.tight_layout()
  
  fig.savefig(os.path.join(fig_directory, 'stage_detection_all_align=%s.pdf'%phase_names[tr]))

### distribution of transition times

dt = np.diff(transitions_cor)

plt.figure(410); plt.clf();
#for i in range(4):
  #plt.subplot(1,5,i+1);
plt.hist(dt[:, :-1], bins = 50, range = [50,200], 
         alpha = 0.45, histtype = 'stepfilled', label = ['L1', 'L2', 'L3', 'L4'], 
         color = ['r', 'orange', 'g', 'blue'])
plt.legend()
plt.xlim(50, 200)



### 24 hr correlation analysis


rate = 3; # Hz sample rate

time_bin = 60; # s;
time_bin = 60 * 5; # s;
time_bin = 60*30;

nbins = rate * time_bin; # number of time bins to calculate mean
ntimes = 5 * 24 * 60 * 60 / time_bin;  # totla number of time steps in analysis
roam_24hr = np.ones((nworms, ntimes)) * np.nan;
r_24hr = np.ones((nworms, ntimes)) * np.nan;
v_24hr = np.ones((nworms, ntimes)) * np.nan;

for wid in range(nworms):  
  # bin speed
  roam = exp.load(strain = 'N2', dtype = 'roam',wid = wid, memmap=None);
  nv = len(roam);
  nvm = nv/nbins * nbins;
  roam_mean = roam[:nvm];
  roam_mean = roam_mean.reshape([nvm/nbins, nbins]);
  roam_mean = np.nanmean(roam_mean, axis = 1);
  roam_mean = np.hstack([roam_mean, [np.nanmean(roam[nvm:])]]);  
    
  ts = int(exp_times[wid] / time_bin);
  roam_24hr[wid, ts:ts+len(roam_mean)] = roam_mean.copy();
  
  
  # bin rotation
  r = exp.load(strain = 'N2', dtype = 'phi',wid = wid, memmap=None);
  nv = len(r);
  nvm = nv/nbins * nbins;
  
  r_mean = r[:nvm];
  r_mean = r_mean.reshape([nvm/nbins, nbins]);
  r_mean = np.nanmean(r_mean, axis = 1);
  r_mean = np.hstack([r_mean, [np.nanmean(r[nvm:])]]);
  
  ts = int(exp_times[wid] / time_bin);
  r_24hr[wid, ts:ts+len(r_mean)] = r_mean.copy();
  
  
  # bin velocity
  v = exp.load(strain = 'N2', dtype = 'xy',wid = wid, memmap=None);
  v = np.linalg.norm(v[3:, :] - v[:-3, :], axis = 1);
  nv = len(v);
  nvm = nv/nbins * nbins;
  
  v_mean = v[:nvm];
  v_mean = v_mean.reshape([nvm/nbins, nbins]);
  v_mean = np.nanmean(v_mean, axis = 1);
  v_mean = np.hstack([v_mean, [np.nanmean(v[nvm:])]]);
  
  ts = int(exp_times[wid] / time_bin);
  v_24hr[wid, ts:ts+len(r_mean)] = v_mean.copy();  

th = np.nanpercentile(roam_24hr, 99);
vplt = roam_24hr.copy();
vplt[vplt > th] = th;
vplt[vplt==0] = np.nan;
plt.figure(700); plt.clf();
plt.imshow(vplt, aspect = 'auto', interpolation = 'none')
plt.colorbar();



phase_names = ['hatch', 'L1', 'L2', 'L3', 'L4', 'L5'];

for tr in range(5):
  fig = plt.figure(500+tr); plt.clf();
  v_24r_algn = np.zeros((v_24hr.shape[0], v_24hr.shape[1]+0));
  tref = np.max(transitions_cor[:, tr]);
  for wid in range(nworms):
    tt = transitions_cor[wid, tr];
    ts =  transitions_cor[wid, 0];
    te =  transitions_cor[wid, -1];      
    nt = te - ts;
    v_24r_algn[wid, ts + tref - tt : ts + tref-tt + nt] = v_24hr[wid,  ts:te];
  
  #rmax = np.nanpercentile(roam_24hr, 95); 
  rmax = np.nanpercentile(v_24hr, 95);
  plt.imshow(v_24r_algn, interpolation = 'none', aspect = 'auto', cmap = plt.cm.viridis, vmax = rmax )
  cb = plt.colorbar(fraction = 0.025, shrink = 0.5, pad = 0.01)
  cb.ax.set_ylabel('rotation [au]', rotation=270, labelpad = 20)
  
  days = np.linspace(0, v_24hr.shape[1] , 6);
  labl = ['%d'%d for d  in 24 *np.linspace(0,5, 6)];  
  plt.xticks(days, labl);
  plt.xlabel('time [hrs]');   
  plt.ylabel('worm id');
  plt.tight_layout()
  
  plt.title(phase_names[tr]);
  plt.tight_layout()
  
  #fig.savefig(os.path.join(fig_directory, 'stage_detection_all_align=%s.pdf'%phase_names[tr]))

# auto correlation in time

import scipy.signal as sig


r_ac = sig.fftconvolve(r_24hr[0], r_24hr[0], mode = 'same')

r_ac = np.ones(r_24hr.shape);
for w in range(nworms):
  r_ac[wid,:] =  sig.fftconvolve(r_24hr[wid], r_24hr[wid], mode = 'same'); 

r_ac_avg = np.nanmean(r_ac, axis = 0)


fig = plt.figure(799); plt.clf();
cc = [];
for wid in range(nworms):
  dd = r_24hr[wid, transitions_cor[wid,0]:transitions_cor[wid,-1]];
  (l, c, a,b) = plt.xcorr(dd, dd, maxlags = 600, usevlines = False, marker = '.');
  cc.append(c);
  
cc_bs = [];
for wid in range(nworms):
  dd = r_24hr[wid, transitions_cor[wid,0]:transitions_cor[wid,-1]];
  (l, c, a,b) = plt.xcorr(dd, np.random.permutation(dd), maxlags = 600, usevlines = False, marker = '.');
  cc_bs.append(c);  
  

fig = plt.figure(799); plt.clf();
for c in cc:
  plt.plot(l,c, 'gray')
plt.plot(l, np.nanmean(np.array(cc), axis = 0), 'r', linewidth = 2)

for c in cc_bs:
  plt.plot(l,c, 'lightgray')
plt.plot(l, np.nanmean(np.array(cc_bs), axis = 0), 'k', linewidth = 2)


hrs = 60 * 60.0 / time_bin
days = 24 * hrs;

ds = [days * d for d in range(-2,3)];
for d in ds:
  plt.plot([d,d], [0,1], 'k')

hs = [hrs * h for h in range(-48,49)]
def lab(i):
  if i % 6 == 0:
    return '%dhr'% i;
  return ''
plt.xticks( hs, [lab(h) for h in range(-48,49)]);
plt.xlabel('time lag')
plt.ylabel('auto-correlation')
plt.tight_layout()
fig.savefig(os.path.join(fig_directory, 'autocorrelation.pdf'))



#


import scipy.signal as sig


v_ac = sig.fftconvolve(v_24hr[0], v_24hr[0], mode = 'same')

v_ac = np.ones(v_24hr.shape);
for w in range(nworms):
  v_ac[wid,:] =  sig.fftconvolve(v_24hr[wid], v_24hr[wid], mode = 'same'); 

v_ac_avg = np.nanmean(r_ac, axis = 0)


fig = plt.figure(798); plt.clf();
cc = [];
for wid in range(nworms):
  dd = v_24hr[wid, transitions_cor[wid,0]:transitions_cor[wid,-1]];
  (l, c, a,b) = plt.xcorr(dd, dd, maxlags = 600, usevlines = False, marker = '.');
  cc.append(c);
  
cc_bs = [];
for wid in range(nworms):
  dd = v_24hr[wid, transitions_cor[wid,0]:transitions_cor[wid,-1]];
  (l, c, a,b) = plt.xcorr(dd, np.random.permutation(dd), maxlags = 600, usevlines = False, marker = '.');
  cc_bs.append(c);  
  

fig = plt.figure(799); plt.clf();
for c in cc:
  plt.plot(l,c, 'gray')
plt.plot(l, np.nanmean(np.array(cc), axis = 0), 'r', linewidth = 2)

#for c in cc_bs:
#  plt.plot(l,c, 'lightgray')
#plt.plot(l, np.nanmean(np.array(cc_bs), axis = 0), 'k', linewidth = 2)


hrs = 60 * 60.0 / time_bin
days = 24 * hrs;

ds = [days * d for d in range(-2,3)];
for d in ds:
  plt.plot([d,d], [0,1], 'k')

hs = [hrs * h for h in range(-48,49)]
def lab(i):
  if i % 6 == 0:
    return '%dhr'% i;
  return ''
plt.xticks( hs, [lab(h) for h in range(-48,49)]);
plt.xlabel('time lag')
plt.ylabel('auto-correlation')
plt.tight_layout()
fig.savefig(os.path.join(fig_directory, 'autocorrelation.pdf'))






import scipy.signal as sig


v_ac = sig.fftconvolve(roam_24hr[0], roam_24hr[0], mode = 'same')

v_ac = np.ones(roam_24hr.shape);
for w in range(nworms):
  v_ac[wid,:] =  sig.fftconvolve(roam_24hr[wid], roam_24hr[wid], mode = 'same'); 

v_ac_avg = np.nanmean(v_ac, axis = 0)


fig = plt.figure(798); plt.clf();
cc = [];
for wid in range(nworms):
  dd = roam_24hr[wid, transitions_cor[wid,0]:transitions_cor[wid,-1]];
  (l, c, a,b) = plt.xcorr(dd, dd, maxlags = 100, usevlines = False, marker = '.');
  cc.append(c);
  
cc_bs = [];
for wid in range(nworms):
  dd = roam_24hr[wid, transitions_cor[wid,0]:transitions_cor[wid,-1]];
  (l, c, a,b) = plt.xcorr(dd, np.random.permutation(dd), maxlags = 100, usevlines = False, marker = '.');
  cc_bs.append(c);  
  

fig = plt.figure(799); plt.clf();
for c in cc:
  plt.plot(l,c, 'gray')
plt.plot(l, np.nanmean(np.array(cc), axis = 0), 'r', linewidth = 2)

#for c in cc_bs:
#  plt.plot(l,c, 'lightgray')
#plt.plot(l, np.nanmean(np.array(cc_bs), axis = 0), 'k', linewidth = 2)


hrs = 60 * 60.0 / time_bin
days = 24 * hrs;

ds = [days * d for d in range(-2,3)];
for d in ds:
  plt.plot([d,d], [0,1], 'k')

hs = [hrs * h for h in range(-48,49)]
def lab(i):
  if i % 6 == 0:
    return '%dhr'% i;
  return ''
plt.xticks( hs, [lab(h) for h in range(-48,49)]);
plt.xlabel('time lag')
plt.ylabel('auto-correlation')
plt.tight_layout()
fig.savefig(os.path.join(fig_directory, 'autocorrelation.pdf'))










# time shuffled data


  
for c in cc:
  plt.plot(l,c, 'gray')
plt.plot(l, np.nanmean(np.array(cc), axis = 0), 'r', linewidth = 2)


# standard auto correlation 


def make_pairs(n):
  p = [];
  for i in range(n):
    for j in range(i+1,n):
      p.append((i,j));
  return p;
           

r_24hr_algn = np.ones((5, r_24hr.shape[0], r_24hr.shape[1]+0)) * np.nan;
for tr in range(5):
  tref = np.max(transitions_cor[:, tr]);
  for wid in range(nworms):
    tt = transitions_cor[wid, tr];
    ts =  transitions_cor[wid, 0];
    te =  transitions_cor[wid, -1];      
    nt = te - ts;
    r_24hr_algn[tr, wid, ts + tref - tt : ts + tref-tt + nt] = r_24hr[wid,  ts:te];

r_mean = np.nanmean(r_24hr, axis = ) 
r_mean_alg = np.nanmean(r_24hr_algn, axis = 1);
plt.figure(802); plt.clf();
plt.plot(r_mean)
plt.plot(r_mean_alg.T)
 
 
 
def corr(data):  
  ns = data.shape[0];
  nt = data.shape[1];
  
  pairs = make_pairs(ns);
  npp = len(pairs);
  
  mean = np.nanmean(data, axis = 0);
  var = np.nanvar(data - mean, axis = 0);
  
  c = np.zeros(nt);
  for p in pairs:
    c += np.nanmean( (data[p[0]] - mean) * (data[p[1]] - mean), axis = 0) / var;
  c /= npp;
  
  return c;
  

r_c = corr(r_24hr);
r_c_alg = [corr(r_24hr_algn[t]) for t in range(5)]

plt.figure(800); plt.clf();
plt.plot(r_c)    
for d in r_c_alg:
  plt.plot(d)


rmax = max(np.nanpercentile(r_24hr, 99.5), np.nanpercentile(r_24hr_algn,99.5))
fig = plt.figure(800); plt.clf();
plt.subplot(3,2,1);
plt.imshow(r_24hr, aspect = 'auto', vmax = rmax, vmin = 0);
plt.title('24hr');
for s in range(5):
  plt.subplot(3,2,s+2);
  plt.imshow(r_24hr_algn[s], aspect = 'auto', vmax = rmax, vmin = 0);
  plt.title(phase_names[s]);
plt.tight_layout();
  
fig.savefig(os.path.join(fig_directory, '24hr_stage_alignment.pdf'))




roam_24hr_algn = np.ones((5, roam_24hr.shape[0], roam_24hr.shape[1]+0)) * np.nan;
for tr in range(5):
  tref = np.max(transitions_cor[:, tr]);
  for wid in range(nworms):
    tt = transitions_cor[wid, tr];
    ts =  transitions_cor[wid, 0];
    te =  transitions_cor[wid, -1];      
    nt = te - ts;
    roam_24hr_algn[tr, wid, ts + tref - tt : ts + tref-tt + nt] = roam_24hr[wid,  ts:te];



rmax = max(np.nanpercentile(roam_24hr, 99.5), np.nanpercentile(roam_24hr_algn,99.5))
fig = plt.figure(800); plt.clf();
plt.subplot(3,2,1);
plt.imshow(roam_24hr, aspect = 'auto', vmax = rmax, vmin = 0);
plt.title('24hr');
for s in range(5):
  plt.subplot(3,2,s+2);
  plt.imshow(roam_24hr_algn[s], aspect = 'auto', vmax = rmax, vmin = 0);
  plt.title(phase_names[s]);
plt.tight_layout();
  
fig.savefig(os.path.join(fig_directory, '24hr_stage_alignment_speed.pdf'))




# absolute time correlation i.e.  <x(t_1) y(t_2)> /
def corr(data):
  ns = data.shape[0];
  nt = data.shape[1];
  
  mean = np.nanmean(data, axis = 0);
  std = np.nanstd(data - mean, axis = 0);
  
  c = np.zeros((nt, nt));
  for t1 in range(nt):
    #for t2 in range(nt):
      #c[t1,t2] = np.nanmean( (data[:,t1] - mean[t1]) * (data[:,t2] - mean[t2]), axis = 0); # / (std[t1] * std[t2]);
    c[t1,:] = np.nanmean( (data[:,:].T * data[:, t1]).T, axis = 0);
  return c;


r_c = corr(r_24hr);
r_c_alg = [corr(r_24hr_algn[t]) for t in range(5)]

p = 5;
vmax = max(np.nanpercentile(r_c, 100-p), np.nanpercentile(r_c_alg, 100-5))
vmin = min(np.nanpercentile(r_c, p), np.nanpercentile(r_c_alg, p))

plt.figure(801); plt.clf();
ax= plt.subplot(3,2,1);
plt.imshow(r_c, aspect = 'auto', vmax = vmax, vmin = vmin, origin = 'lower');
plt.colorbar()
plt.title('24hr');
for s in range(5):
  plt.subplot(3,2,s+2, sharex = ax, sharey = ax);
  plt.imshow(r_c_alg[s], aspect = 'auto', vmax = vmax, vmin = vmin, origin = 'lower');
  plt.title(phase_names[s]);
  plt.colorbar()
plt.tight_layout();







### correlation with time absolute and relative

r_24hr_algn = np.ones((6, r_24hr.shape[0], r_24hr.shape[1]+0)) * np.nan;
r_24hr_algn[0] = r_24hr.copy();
for tr in range(5):
  tref = np.max(transitions_cor[:, tr]);
  for wid in range(nworms):
    tt = transitions_cor[wid, tr];
    ts =  transitions_cor[wid, 0];
    te =  transitions_cor[wid, -1];      
    nt = te - ts;
    r_24hr_algn[tr+1, wid, ts + tref - tt : ts + tref-tt + nt] = r_24hr[wid,  ts:te];



def corr_t(data):
  mean = np.nanmean(data);
  return np.nanmean(data - mean, axis = 0);
  
cc = np.array([corr_t(x) for x in r_24hr_algn]);

plt.figure(801); plt.clf();
names = ['24hr', 'H', 'L1', 'L2', 'L3', 'L4', 'L5']
vmax = np.nanpercentile(cc, 95);
vmin = np.nanpercentile(cc,  5);

for s,c in enumerate(cc):
  if s == 0:
    ax= plt.subplot(3,2,s+1);
    #plt.imshow(r_c, aspect = 'auto', vmax = vmax, vmin = vmin, origin = 'lower');
  else:
    plt.subplot(3,2,s+1, sharex = ax, sharey = ax);
  #plt.imshow(c, aspect = 'auto', vmax = vmax, vmin = vmin, origin = 'lower');
  plt.plot(c);
  plt.title(phase_names[s]);
  #plt.colorbar()
plt.tight_layout();

plt.figure(802); plt.clf();
plt.plot(cc.T)





# mean activity with shuffeled starting times


trj = [];
for wid in range(nworms):
  ts =  transitions_cor[wid, 0];
  te =  transitions_cor[wid, -1];      
  nt = te - ts;
  trj.append(r_24hr[wid,  ts:te]); 
  
nsamp = 5000;
r_24hr_shuffle = np.ones((nsamp, r_24hr.shape[0], r_24hr.shape[1]+0)) * np.nan;
for s in range(nsamp):
  o = np.random.permutation(nworms);
  for w in range(nworms):
    ts =  transitions_cor[o[w], 0];
    te = min(r_24hr.shape[1], ts+len(trj[w]));
    r_24hr_shuffle[s, w, ts:te] = trj[w][:te-ts];
 
def corr_t(data):
  mean = np.nanmean(data);
  return np.nanmean(data - mean, axis = 0);
  
cc_shuffle = np.array([corr_t(x) for x in r_24hr_shuffle]);
 
cc_shuffle_max = np.nanmax(cc_shuffle, axis = 0);
cc_shuffle_min = np.nanmin(cc_shuffle, axis = 0);
 
fig = plt.figure(803); plt.clf();
#plt.plot(cc_shuffle.T, 'gray');
plt.plot(cc_shuffle_max, 'gray');
plt.plot(cc_shuffle_min, 'gray');

cc = np.array([corr_t(x) for x in r_24hr_algn]);

names = ['24hr', 'H', 'L1', 'L2', 'L3', 'L4', 'L5']
vmax = np.nanpercentile(cc, 95);
vmin = np.nanpercentile(cc,  5);

plt.plot(cc[0], 'r')
plt.xlabel('time')
plt.ylabel('correlation with time')



### shuffled transition times


trj = [];
for wid in range(nworms):
  ts =  transitions_cor[wid, 0];
  te =  transitions_cor[wid, -1];      
  nt = te - ts;
  trj.append(r_24hr[wid,  ts:te]); 

names = ['H', 'L1', 'L2', 'L3', 'L4', '24hr'] 
nsamp = 50;

transitions_cor_0 = np.zeros((transitions_cor.shape[0], 6));
transitions_cor_0 = transitions_cor.copy();
transitions_cor_0[:,-1] = 0;


tr_avg = np.array(np.mean(transitions_cor_0, axis = 0), dtype = int)

def corr_t(data):
  mean = np.nanmean(data);
  return np.nanmean(data - mean, axis = 0);

fig = plt.figure(803); plt.clf();

for tr in range(6):
  r_24hr_shuffle = np.ones((nsamp, r_24hr.shape[0], r_24hr.shape[1]+0)) * np.nan;
  for s in range(nsamp):
    if s == 0:
      ref = np.zeros(nworms, dtype = int);
    else:
      ref = np.random.permutation(nworms);
    
    for w in range(nworms):
      to = transitions_cor_0[ref[w], tr];
      to0 = transitions_cor_0[ref[w], 0];
      tw = transitions_cor_0[w, tr];
      tw0 = transitions_cor_0[w,0];
      
      ts = int(max(0, t0 + to - tw - (to0 - tw0)));
      te = int(min(r_24hr.shape[1], ts+len(trj[w])));
      r_24hr_shuffle[s, w, ts:te] = trj[w][:te-ts];
      
    #plt.figure(8); plt.clf(); plt.imshow(r_24hr_shuffle[0], aspect = 'auto')
  
  cc_shuffle = np.array([corr_t(x) for x in r_24hr_shuffle]);
   
  cc_shuffle_max = np.nanmax(cc_shuffle[1:], axis = 0);
  cc_shuffle_min = np.nanmin(cc_shuffle[1:], axis = 0);
  
  plt.subplot(3,2,tr+1);
  plt.plot(cc_shuffle_max, 'gray');
  plt.plot(cc_shuffle_min, 'gray');
  plt.plot(cc_shuffle[0], 'r')
  
  plt.xlabel('time')
  plt.ylabel('correlation with time')
  plt.ylim(-0.8, 1.0);
  plt.xlim(0, r_24hr.shape[1]);
  plt.title(names[tr]);
  






# calculate correlation at absolute time





# calcualte correlation at algined traces 



# auto correlation

sig.




