# -*- coding: utf-8 -*-
"""
Compare manual and automatic detected life stage durations

@author: ckirst
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import copy

dir_roaming    = '/home/ckirst/Science/Projects/CElegans/Analysis/Roaming/Code'
dir_behaviour  = '/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Code/'


os.chdir(dir_roaming);
import experiment as rexp

os.chdir(dir_behaviour);
import analysis.experiment as exp

fig_directory = '/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Figures/StageDetection/'

import scripts.preprocessing.filenames as fn
strain = 'tph1'
nworms, exp_names, dir_names = fn.filenames(strain);

import analysis.plot as fplt;

import signalprocessing.peak_detection as pd
import scipy.signal as sig

def average(x, nbins):
  """Bin data using nbins windows"""
  n = len(x); 
  nm = n/nbins * nbins;
  x_mean = x[:nm];
  x_mean = x_mean.reshape([nm/nbins, nbins]);
  x_mean = np.nanmean(x_mean, axis = 1);
  x_mean = np.hstack([x_mean, [np.nanmean(x[nm:])]]);
  return x_mean;


#%% Parameter  
rate = 3; # Hz sample rate
time_bin = 5*60; # [sec] for averaging 

# derived 
nbins = rate * time_bin; # number of time bins to calculate mean

### Create Data Matrices

r_all_0 = [];
nmax = 0;
for wid in range(nworms):
  r = exp.load(strain = strain, wid = wid, dtype = 'rotation')
  r_mean = average(r, nbins);
  r_all_0.append(r_mean);
  nmax = max(nmax, len(r_mean));

r_all = np.zeros((nworms, nmax));
for wid in range(nworms):
  r_all[wid,:len(r_all_0[wid])] = r_all_0[wid];


v_all_0 = [];
nmax = 0;
for wid in range(nworms):
  v = exp.load(strain = strain, wid = wid, dtype = 'speed')
  v_mean = average(v, nbins);
  v_all_0.append(v_mean);
  nmax = max(nmax, len(v_mean));

v_all = np.zeros((nworms, nmax));
for wid in range(nworms):
  v_all[wid,:len(v_all_0[wid])] = v_all_0[wid];



lv = 0; lr = 0;
for wid in range(nworms):
  xy = exp.load(strain = strain, dtype = 'speed', wid = wid);
  lv = max(lv, xy.shape[0]);
  r = exp.load(strain = strain, dtype = 'rotation', wid = wid);
  lr = max(lr, r.shape[0]);

v_full = np.zeros((nworms, lv));
r_full = np.zeros((nworms, lr));
for wid in range(nworms):
  v = exp.load(strain = strain, dtype = 'speed', wid = wid, memmap = None);
  nv = len(v);
  v_full[wid,:nv] = v;
  
  r = exp.load(strain = strain, dtype = 'rotation', wid = wid, memmap = None);
  nr = r.shape[0];
  r_full[wid,:nr] = r;



#%% Determine Lethargus phases

l_pks_0 = [];
k = 0;
verbose = True;

for wid in range(nworms):
  r = exp.load(strain = strain, wid = wid, dtype = 'rotation')
  r_mean = average(r, nbins);
  
  # smooth roaming curve
  rf = sig.savgol_filter(r_mean, 51, 7)

  # find minima and mark
  pks = pd.find_peaks(rf, 0.5);
  l_pks_0.append(pks);
  
  if verbose: 
    if wid % 16 == 0:
      k += 1;
      fig = plt.figure(111+k); plt.clf();
      plt.subplot(4,4,1);
      i = 2;
    else:
      plt.subplot(4,4,i);
      i += 1;
    plt.plot(r_mean, '.')  
    plt.plot(rf, 'k')
    plt.title('worm %d' % wid);
    plt.scatter(pks[:,0], pks[:,1], s = 60, c = 'r')
  


#%% Remove and add peaks manually


l_pks = copy.copy(l_pks_0)

pks_rmv = {4 : [4],  8: [4], 10 : [4,5], 11 : [6], 12 : [5,6], 17 : [5], 30 : [4,5], 33 : [5] }
pks_add = {4: [170], 8 : [416],  10 : [475], 18 : [293], 22 : [290], 30 : [470] }

for k,v in pks_rmv.iteritems():
  l_pks[k]  = np.delete(l_pks[k], v, axis = 0);
  
for k,v in pks_add.iteritems():
  l_pks[k] = np.vstack((l_pks[k], np.vstack((v, np.ones(len(v)))).T));


### Remove initial peaks and final ones
for i in range(len(l_pks)):
  t = l_pks[i][:,0];
  l_pks[i] = l_pks[i][np.argsort(t),:];
  
  #l_pks[i] = l_pks[i][np.sort(t) <= 620,:];
  
  if len(l_pks[i]) > 4:
    l_pks[i] = l_pks[i][-4:,:];


### Check Detected Peaks
if verbose: 
  k = 0;
  for wid in range(nworms):    
    r = exp.load(strain = strain, wid = wid, dtype = 'rotation')
    r_mean = average(r, nbins);
  
    # smooth roaming curve
    rf = sig.savgol_filter(r_mean, 51, 7)
  
    if wid % 16 == 0:
      k += 1;
      fig = plt.figure(111+k); plt.clf();
      plt.subplot(4,4,1);
      i = 2;
    else:
      plt.subplot(4,4,i);
      i += 1;
    plt.plot(r_mean, '.')  
    plt.plot(rf, 'k')
    plt.title('worm %d' % wid);
    plt.scatter(l_pks[wid][:,0], l_pks[wid][:,1], s = 60, c = 'r')


lethargus = [];
for wid in range(nworms):
  t = l_pks[wid][:,0];
  if len(t) != 4:
    print wid, len(t)
    assert False
  lethargus.append(t);
  
lethargus = np.abs(np.array(lethargus))


#%% Hatching 

hatching = np.array([np.nonzero(r_all[wid, :] > 0.5)[0][0] for wid in range(nworms)])
  
# jump to next onset
for wid in range(nworms):
  gap = np.nonzero(np.isnan(r_all[wid][hatching[wid]:]))[0];
  #gap = np.nonzero(roam_24hr[wid][hatching[wid]:] == 0.0)[0];  
  
  if len(gap) > 0:
    if gap[0] < 20:
      gap = np.append(gap, gap[-1]+10);
      dgap = np.diff(gap);
      hatching[wid] = hatching[wid] + gap[0] + np.nonzero(dgap > 5)[0][0] + 1;


final = np.array([np.nonzero(r_all[wid, :] > 0.0)[0][-1] for wid in range(nworms)])

#%% Hatching Velocity Onset

v_all_plt = v_all.copy();
v_th = np.nanpercentile(v_all, 80);
v_all_plt[v_all_plt > v_th] = v_th;

plt.figure(1); plt.clf();
plt.imshow(v_all_plt, aspect = 'auto')

plt.figure(2); plt.clf();
plt.plot(v_all[9])

# 0.4 seems to be a good velocity threshold for hatching 

# add shift

add_shift = {0 : 10, 1 : 10, 2 : 10, 3 : 10, 5: 10, 14 : 1, 15 : 1, 16: 10, 17 : 5, 18: 10, 20 : 10, 21 : 6, 22 : 6, 37 : 4, 47 : 10};
             
hatching_cor = hatching.copy();
for k,v in add_shift.iteritems():
  hatching_cor[k] += v;

h = np.array([np.nonzero(v_all[wid,hatching_cor[wid]:] > 0.45)[0][0] for wid in range(nworms)]);

hatching_cor += h - 1;
#transitions_cor[27,0] += -12;


plt.figure(3); plt.clf();
plt.imshow(v_all_plt, aspect = 'auto', origin = 'lower')
plt.scatter(hatching_cor, range(nworms), s = 60, c = 'r')

wid = 35
plt.figure(2); plt.clf();
plt.plot(v_all[wid])
plt.scatter(hatching_cor[wid], v_all[wid][hatching_cor[wid]], c ='r', s = 60)


#%% Compose Transitions Indices

transitions = np.zeros((nworms, 6), dtype = int)

for wid in range(nworms):
  transitions[wid, 0] = hatching_cor[wid];
  transitions[wid, 1:-1] = lethargus[wid];
  transitions[wid, -1] = final[wid];
  transitions[wid, :] = np.sort(transitions[wid,:])

transitions[transitions < 0] = 0; 
transitions[:,-1] = transitions[:,-2] + int(16.0 * 60 * 60 / time_bin);

### Plot the estimated transition times

fig = plt.figure(300); plt.clf();
rmax = np.nanpercentile(r_all, 95);
plt.imshow(r_all, interpolation = 'none', aspect = 'auto', cmap = plt.cm.viridis, vmax = rmax )

cb = plt.colorbar(fraction = 0.025, shrink = 0.5, pad = 0.01)
cb.ax.set_ylabel('rotation [au]', rotation=270, labelpad = 20)

xyt = np.zeros((nworms*6, 2)); 
for wid in range(nworms):
  xyt[wid*6:(wid+1)*6, 0] = transitions[wid];
  xyt[wid*6:(wid+1)*6, 1] = wid;
  
plt.scatter(xyt[:,0],xyt[:,1], c = 'r', s = 20);
plt.xlim(-0.5, r_all.shape[1]-0.5);
plt.ylim(-0.5, nworms-0.5);

days = np.linspace(0, r_all.shape[1] , 6);
labl = ['%d'%d for d  in 24 *np.linspace(0,5, 6)];
plt.xticks(days, labl);
plt.xlabel('time [hrs]'); 
plt.ylabel('worm id');
plt.tight_layout()

fig.savefig(os.path.join(fig_directory, '%s_stage_detection_all.pdf' % strain))



#%% Plot example figure for lethragus phase detection

fig = plt.figure(106); plt.clf();

wid = 0;
rf = sig.savgol_filter(r_all[wid, :],51, 7)
plt.scatter(l_pks[wid][:,0], l_pks[wid][:,1], s = 60, c = 'r')
plt.plot(rf, 'k')
plt.plot(r_all[wid,:], '.')  

days = np.linspace(0, r_all.shape[1] , 6);
labl = ['%d'%d for d  in 24 *np.linspace(0,5, 6)];
plt.xticks(days, labl);
plt.xlabel('time [hrs]'); 
plt.ylabel('-rotation');

fig.savefig(os.path.join(fig_directory, '%s_stage_detection_wid=%d.pdf'% (strain, wid)))





#%% Transition times - high time resolution

def binned_average(x, bin_width = 10):
  n = len(x);
  ne = int(np.ceil(n*1.0/bin_width)) * bin_width;
  xx = np.ones(ne);
  xx[:n] = x;
  x_mean = xx.reshape([ne/bin_width, bin_width]);
  x_mean = np.nanmean(x_mean, axis = 1);
  return x_mean;

def find_min(x, bin_width = 10):
  xm = binned_average(x, bin_width=bin_width);
  imin = np.nanargmin(xm);
  return int((imin + 0.5) * bin_width);

def find_transitions(x, transition0, window = 500, bin_width = 10):
  n = len(x);
  pos = np.zeros(len(transition0));
  for i,t in enumerate(transition0):
    ts = max(0,t-window);
    te = min(n,t+window);
    #print ts,te,n
    pos[i] = find_min(x[ts:te], bin_width = bin_width) + ts;
  return pos;
    


#%% Transitions times as indices in original data 

transitions_id = np.array([rate * time_bin * transitions[w,:] for w in range(nworms)])
transitions_id[:,-1] = transitions_id[:,-2] + 16 * 60 * 60 * rate; # 16hrs into adulthood
transitions_id[transitions_id < 0] = 0;

transitions_id_fine = transitions_id.copy();
#for wid in range(nworms):
#  #r = exp.load(strain = 'n2', wid = wid, dtype = 'rotation')
#  v = exp.load(strain = strain, wid = wid, dtype = 'speed')
#  #transitions_id_fine[wid][1:-1] = find_transitions(v, transitions_id[wid][1:-1], window = 20000, bin_width = 3 * 60)
#  transitions_id_fine[wid][1:-1] = find_transitions(v, transitions_id[wid][1:-1], window = 10000, bin_width = 3 * 60 * 15)

# not fine tuning really required !

#%% Plot original data with transition points

r_full_plt = r_full.copy();
r_full_plt[np.isnan(r_full)] = 0;
r_th = np.percentile(r_full_plt.flat, [95]);
r_full_plt[r_full_plt > r_th] = r_th;
for wid in range(nworms):
  for s in range(6):
    r_full_plt[wid, max(0, transitions_id_fine[wid,s]-200):min(r_full_plt.shape[1], transitions_id_fine[wid,s]+200)] = 1.2*r_th
fplt.plot_array(r_full_plt)

v_full_plt = v_full.copy();
v_full_plt[np.isnan(v_full)] = 0;
v_th = np.percentile(v_full_plt.flat, [95]);
v_full_plt[v_full_plt > v_th] = v_th;
for wid in range(nworms):
  for s in range(6):
    v_full_plt[wid, max(0, transitions_id_fine[wid,s]-200):min(v_full_plt.shape[1], transitions_id_fine[wid,s]+200)] = 1.2*v_th
fplt.plot_array(v_full_plt)



#%%############################################################################
### Save data

#np.save(os.path.join(exp.data_directory, 'n2_stage.npy'), transitions_id_fine)

### Add length to the data set

#st = np.load(os.path.join(exp.data_directory, 'n2_stage.npy'))

l = [];
for wid in range(nworms):
  print wid;
  xy = exp.load(strain = strain, wid = wid);
  l.append(len(xy));

st = np.concatenate((transitions_id_fine, np.array([l]).T), axis = 1)

np.save(os.path.join(exp.data_directory, '%s_stage.npy' % strain), st)


#%% Make sure the final stage time is within data set 

st = exp.load_stage_times(strain = strain);

ids = np.where(st[:,-1] < st[:,-2])[0];
if len(ids) > 0:
  st[ids,-2] = st[ids, -1];

np.save(os.path.join(exp.data_directory, '%s_stage.npy' % strain), st)









#%%############################################################################
### Analyse Transitions



#%% Align to different transitions

def plot_aligned(data_all, transitions, label = 'data', dt = time_bin, phase = 0):
  phase_names = ['hatch', 'L1', 'L2', 'L3', 'L4', 'L5'];
  
  ttref0 = np.max(transitions[:, 0]);
  ntt = data_all.shape[1] + ttref0 + 500;
  nw = data_all.shape[0];

  tr = phase;
  data_algn = np.zeros((nw, ntt));
  tref = np.max(transitions[:, tr]);
  for wid in range(nworms):
    tt = transitions[wid, tr];
    ts =  transitions[wid, 0];
    te =  transitions[wid, -1];      
    nt = te - ts;
    t0 = ts + tref - tt;
    t1 = t0 + nt;
    #print 'tref=%s, ts =%d, te=%d, tt =%d, nt = %d, t0=%d, t1=%d, ntt=%d' % (tref, ts,te,tt, nt, t0,t1, ntt)
    data_algn[wid, t0 : t1] = data_all[wid,  ts:te];
    
  #rmax = np.nanpercentile(roam_24hr, 95); 
  rmax = 1;
  plt.imshow(data_algn, interpolation = 'none', aspect = 'auto', cmap = plt.cm.viridis, vmax = rmax )
  cb = plt.colorbar(fraction = 0.025, shrink = 0.5, pad = 0.01)
  cb.ax.set_ylabel(label + ' [au]', rotation=270, labelpad = 20)
  
  days = np.array(24. * 60 * 60 / time_bin * np.arange(6), dtype = int);
  labl = ['%d'%d for d  in 24 *np.linspace(0,5, 6)];  
  plt.xticks(days, labl);
  plt.xlabel('time [hrs]');   
  plt.ylabel('worm id');
  plt.tight_layout()
  
  plt.title(phase_names[tr]);
  plt.tight_layout()
    
    
for p in range(5):
  fig = plt.figure(400+p); plt.clf();
  plot_aligned(r_all, transitions, dt = time_bin, phase = p)
  
  phase_names = ['hatch', 'L1', 'L2', 'L3', 'L4', 'L5'];
  fig.savefig(os.path.join(fig_directory, 'stage_detection_all_align=%s.pdf' % phase_names[p]))

### Distribution of transition times

dt = np.diff(transitions_id_fine) * 1.0/rate / 60 / 60

fig = plt.figure(410); plt.clf();
for s in range(4):
  plt.subplot(4,1,s+1);
  #for i in range(4):
    #plt.subplot(1,5,i+1);
  plt.hist(dt[:, s], bins = 40, range = [5,15], 
         alpha = 0.45, histtype = 'stepfilled', label = ['L1', 'L2', 'L3', 'L4'][s], 
         color = ['r', 'orange', 'g', 'blue'][s])
  plt.legend()
  plt.xlim(5, 15)

fig.savefig(os.path.join(fig_directory, 'stage_durations_histogram.pdf'))


#%% Correlation between transitions times

fig = plt.figure(411); plt.clf();
k = 0;
for i in range(4):
  for j in range(i):
    k+=1;
    plt.subplot(2,3,k);
    plt.plot(dt[:,j], dt[:,i], '.');
    plt.title('L%d vs L%d' % (j+1,i+1))


fig.savefig(os.path.join(fig_directory, 'stage_durations_correlation.pdf'))



#%% PCA on stage durations

fig = plt.figure(412); plt.clf();
fplt.plot_pca(dt[:,:-1])

fig.savefig(os.path.join(fig_directory, 'stage_durations_pca.pdf'))






#%% Get stage durations from Roaming Dwelling data set

strain = 'N2';

rd_data = rexp.load_data(strain);

rd_stage_ids = rd_data.stage_switch;
rd_stage_dur = rd_data.stage_durations;

#%% Get stage durations from automatic detection

xy_stage_ids = np.load(os.path.join(exp.data_directory, 'transitions_times.npy'))
xy_stage_dur = np.diff(xy_stage_ids, axis = 1)


#%% Plot Stage Durations 


fig = plt.figure(1); plt.clf();
rate = 3.0 * 60 * 60;
rate_unit = 'hr'

for s in range(4):
  plt.subplot(2,2, s+1);
  plt.plot(xy_stage_dur[:, s]/rate, rd_stage_dur[:, s]/rate,'*')
  plt.title('L%d' % (s+1));
  plt.xlabel('t ['+rate_unit +'] auto');
  plt.ylabel('t ['+rate_unit+ '] hand');
  
  xmm = [f(xy_stage_dur[:, s]/rate) for f in [np.min, np.max]];
  ymm = [f(rd_stage_dur[:, s]/rate) for f in [np.min, np.max]];
  plt.plot(xmm,ymm, 'r')
  plt.xlim(xmm[0]-0.5, xmm[1]+0.5)
  plt.ylim(ymm[0]-0.5, ymm[1]+0.5)

fig.savefig(os.path.join(fig_directory, 'stage_durations_auto_vs_hand.pdf'))




#%% Stage Durations rounded

rate = 3.0 * 60 * 60;

xy_switch_hr = np.round((xy_stage_ids.T - xy_stage_ids[:,0]).T / rate);
rd_switch_hr = np.round(rd_stage_ids / rate);

#xy_switch_hr = xy_switch / rate
#rd_switch_hr = rd_switch / rate

fig = plt.figure(2); plt.clf();
plt.plot(xy_switch_hr[:,:-1], rd_switch_hr[:,:-1],'*')
plt.plot(np.linspace(0, 50, 10), np.linspace(0,50,10), 'k')
plt.title('L%d' % (s+1));
plt.xlabel('t [hr] (automatic)');
plt.ylabel('t [hr] (manual)');
plt.grid(True)
fig.savefig(os.path.join(fig_directory, '%s_stage_durations_auto_vs_hand_rounded.pdf' % strain))


import scipy.stats as st

lr = st.linregress(xy_switch_hr[:,:-1].flatten(), rd_switch_hr[:,:-1].flatten())




#%% Roaming data


rd_all_0 = [];
nmax = 0;
for wid in range(nworms):
  rd = exp.load(strain = 'n2', wid = wid, dtype = 'roam')
  rd_mean = average(rd, nbins);
  rd_all_0.append(rd_mean);
  nmax = max(nmax, len(rd_mean));

rd_all = np.zeros((nworms, nmax));
for wid in range(nworms):
  rd_all[wid,:len(rd_all_0[wid])] = rd_all_0[wid];


plt.figure(200); plt.clf();
plt.imshow(rd_all, aspect ='auto')

tfinal = transitions[:,-1].max()
for p in range(4):
  fig = plt.figure(201+p); plt.clf();
  plot_aligned(rd_all[:,:tfinal], transitions, phase = p)
  fig.savefig(os.path.join(fig_directory, 'stage_detection_roaming_align=%s.pdf' % phase_names[p]))
  plt.xlim(0, 72 * 60 * 60 / time_bin)

