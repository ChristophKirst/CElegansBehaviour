# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:36:27 2017

@author: ckirst
"""

import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);

#%% test WromPlot


wids = np.array([39, 9, 26, 96, 48, 87]) - 1;


reload(v);
fig = plt.figure(2); plt.clf();
ax = plt.subplot(1,1,1);

i = 0;
w = v.WormPlot(ax, strain = 'n2', wid = wids[i], time = t+i*300,
           time_stamp_offset= 34200,
           trajectory_length=150, trajectory_delta=30, trajectory_size=3, 
           trajectory_cmap= plt.cm.gray_r,
           stage_stamp_font_size=12,
           stage_indicator_label_font_size=12,
           time_stamp = False,
           time_stamp_font_size=12,
           border = [[5,5],[50,5]])
a = v.WormAnimation(figure = fig, plots = [w], times = [range(w.stage_time_min + 1500*300, w.stage_time_max, 300)]);

fig.show()


#%%
a.animate()



#%% All worms

wids = np.arange(124);
wids[90] = 124;  # wid = 90 is corrupted data !

# sort via average roaming fraction
roaming = exp.load(strain = 'n2', wid = wids, dtype = 'roam');


roaming_mean = np.array([np.nanmean(r) for r in roaming]);

rorder = np.argsort(roaming_mean);

wids = np.array(wids)[rorder];


#%% persitence based ordering

wids = np.array(np.genfromtxt('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Data/persistance_median_order.csv'), dtype = int);
wids = wids - 1;


#%%

dwell = np.array([55, 72, 49, 51, 5, 69, 45, 73, 84, 95]) - 1;
roam  = np.array([23, 79, 123, 87, 48, 96, 26, 39, 9]) - 1;

stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
duration = np.sum(np.diff(stimes, axis = 1), axis = 1);
duration_min = np.min(duration);
dtmin = 1;

stimes[1,0] = 60000;

stimes[:,-1] = stimes[:,0] + duration_min;
#stimes[:,0] = stimes[:,-1] - dtmin;

#stimes[:,-1] = stimes[:,0] + dtmin;
#stimes[:,0]-stimes[:,-1]


dt = 250;
sp = 10;
traj_hist = 120*60 / dt * sp;

times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)


fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_life_persisters_median.mov');
#fname = None;

#%%

reload(v);
fig = plt.figure(1); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.015, hspace = 0.075, wspace = 0.075)

def stage_stamp_position(xlim, ylim):
  return [xlim[1] + 60, ylim[0] + 30];

plots = [];
for wi,w in enumerate(wids):
  ax = plt.subplot(9,14,wi+1);
  tcm = plt.cm.gray_r;
  #pcol = 'black'
  #if w in dwell:
  #  tcm = plt.cm.Reds;
  #  pcol = 'red';
  #elif w in roam:
  #  tcm = plt.cm.Blues;
  #  pcol = 'blue';
  pcol = plt.cm.coolwarm(1.0 * wi / len(wids));  
  
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[wi,0],   
           image = False,
           
           plate_color = pcol,

           trajectory = True,
           trajectory_length=150, trajectory_delta= dt/sp, trajectory_size=1.5, 
           trajectory_cmap= tcm,

           stage_stamp_font_size=8,
           
           stage_indicator = False,
           stage_indicator_label_font_size=10,
           
           stage_stamp = True,
           stage_stamp_position = stage_stamp_position,   
           
           time_stamp = False,
           time_stamp_font_size=12,
           time_stamp_offset= 34200,           
           
           border = [[10,10],[10,10]]);       
  plots.append(p);



#%%

def time_text(time):
  return 'N2 %s' % v.time_str(time);

a = v.WormAnimation(figure = fig, plots = plots, times = times,
                    time_stamp = True,
                    time_stamp_text = time_text,
                    time_stamp_font_size=10,
                    save = fname, fps = 30);

fig.show()



#%%
a.animate()












#%%


import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);



#%% Animate single worms


#wids = np.array([39, 9, 26, 96, 48, 87]) - 1;
wids = np.array([68, 3, 36, 77, 98, 92, 88, 18, 29, 125, 35]) - 1;


i = 2;
wid = wids[i];

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_%d_life.mov' % wid);

stimes = exp.load_stage_times(strain = 'n2', wid = wid);


stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];

dt = 10 * 3; # min * sec/min * sample rate
fps = 60;
subsample = 10;
traj_hist = 5 # history of traj in minutes
traj_length = traj_hist * 3 * 60 * subsample / dt;
print 'trajectory history is %f min, sampled every %f secs = %d points' % ((1.0 * traj_length * dt / subsample / 3.0 / 60), 1.0 * dt /subsample /3, traj_length) ;
print 'frame time difference: %f sec' % (dt / 3.0)


reload(v);
fig = plt.figure(2, figsize = (2,2), dpi = 360, facecolor = 'w'); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.035, hspace = 0.075, wspace = 0.075)
ax = plt.subplot(1,1,1);

times=  np.array([range(stimes[0],stimes[-2],dt)]);
#tt = times[times > 492127 ]
tt = 492420;
tt = times[0,-200]

import skimage.morphology as morph
import skimage.measure as meas

def image_filter(img):
  img2 = img.copy();
  img2[img2 < 30] = 100;
  img2 = exp.smooth_image(img2, sigma = 1.0);
  #plt.figure(6); plt.clf();
  #plt.imshow(img2);

  # threshold image and take zero smaller components..
  
  th = img2 < 92;

  th2 = morph.binary_closing(th, morph.diamond(1))
  
  label = meas.label(th2, background=0)
  #plt.imshow(mask)
  
  bs = meas.regionprops(label+1);
  area = np.array([prop.area for prop in bs]);
  if len(area) > 0:
    mask = np.logical_and(label > -1, label != np.argsort(area)[-1]);
  
    img2[mask] = 100;
  
  img2[:2,:] = 100; img2[-2:,:] = 100;
  img2[:,:2] = 100; img2[:,-2:] = 100;
  
  #plt.figure(6); plt.clf();
  #plt.subplot(1,2,1);
  #plt.imshow(img2, vmin = 84, vmax = 92, cmap = plt.cm.gray)
  #plt.subplot(1,2,2);
  #plt.imshow(img2);
  return img2;

w = v.WormPlot(ax, strain = 'n2', wid = wid, time = stimes[0], # time = tt,#
               image_filter=image_filter, image_sigma = None, image_vmin = 84, image_vmax = 92,
               time_stamp_offset= stimes[0],
               trajectory_length = traj_length, trajectory_delta = dt/subsample, trajectory_size = 0.5, 
               trajectory_cmap= plt.cm.Reds,
               stage_stamp_font_size = 6, stage_stamp_colors= stage_colors,
               stage_indicator_label_font_size = 5.5, stage_indicator_colors=stage_colors,
               stage_indicator_label_offset = 55,
               time_stamp = True,
               time_stamp_font_size=6, 
               plate_linewidth = 0.5,     
               border = [[5,5],[5,100]])
a = v.WormAnimation(figure = fig, plots = [w], times = np.array([range(stimes[0],stimes[-2],dt)]),
                    fps = fps, dpi = 360, save = fname, time_stamp = None);

fig.show()

#%%

a.animate()





#%%











#%%

import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);


#%% Gui

reload(v)
g = v.WormGui(strain = 'n2', wid = 35, time = 400000, trajectory_length = 3*60*60 / 30, trajectory_delta=30);



#%%

movie_times = [29323 - 200 , 173484 - 1540, 273438 - 540, 370951 - 540, 526599 - 540];

movie_times =np.array( [69867 + 3*60*30, 169200 + 3*60*60, 275400 + 3 * 60 * 60 - 2000  - 3*3*60, 406799 + 3*60*60, 503927 + 3*60*60 ]) -3*60*60

#%% Animate single worms high time resolution


#wids = np.array([39, 9, 26, 96, 48, 87]) - 1;
wids = np.array([68, 3, 36, 77, 98, 92, 88, 18, 29, 125, 35]) - 1;
wid = 35;

stage = 0;
#stage_offs = [210,0,0,0,0]; # in min
#stage_offs = np.array(stage_offs, dtype = int) * 3 * 60;


fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_%d_life_stage=%s.mov' % (wid, stage));

stimes = exp.load_stage_times(strain = 'n2', wid = wid);


stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];

dt = 3; #  sec * sample rate
fps = 30;
subsample = 1;
traj_hist = 30 # history of traj in minutes
traj_length = traj_hist * 3 * 60 * subsample / dt;
print 'trajectory history is %fmin' % (1.0 * traj_length * dt / subsample / 3.0 / 60);

reload(v);
fig = plt.figure(2, figsize = (2,2), dpi = 360, facecolor = 'w'); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.035, hspace = 0.075, wspace = 0.075)
ax = plt.subplot(1,1,1);

xy = exp.load(strain = 'n2', wid = wid, dtype = 'xy');
xy[:movie_times[stage]-1,:] = np.nan;

w = v.WormPlot(ax, strain = 'n2', wid = wid, time = movie_times[stage], xy = xy,
           time_stamp_offset= stimes[0],
           trajectory_length = traj_length, trajectory_delta = dt/subsample, trajectory_size = 0.1, 
           trajectory_cmap= plt.cm.Reds,
           stage_stamp_font_size = 6, stage_stamp_colors= stage_colors,
           stage_indicator_label_font_size = 5.5, stage_indicator_colors=stage_colors,
           time_stamp = True,
           time_stamp_font_size=6, 
           plate_linewidth = 0.5,     
           border = [[5,5],[5,100]])
           
#%%
a = v.WormAnimation(figure = fig, plots = [w], times = np.array([range(movie_times[stage], movie_times[stage]+3*60*traj_hist,dt)]),
                    fps = fps, dpi = 360, save = fname, time_stamp = None);

fig.show()

#%%

a.animate()






















#%%


import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);



#%% All worms

wids = np.arange(124);
wids[90] = 124;  # wid = 90 is corrupted data !

# sort via average roaming fraction
roaming = exp.load(strain = 'n2', wid = wids, dtype = 'roam');


roaming_mean = np.array([np.nanmean(r) for r in roaming]);

rorder = np.argsort(roaming_mean);

wids = np.array(wids)[rorder];


#%% persitence based ordering

wids = np.array(np.genfromtxt('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Data/persistance_median_order.csv'), dtype = int);
wids = wids - 1;


#%%

dwell = np.array([55, 72, 49, 51, 5, 69, 45, 73, 84, 95]) - 1;
roam  = np.array([23, 79, 123, 87, 48, 96, 26, 39, 9]) - 1;

wids_large = np.array([39, 51], dtype = int) - 1;

stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
duration = np.sum(np.diff(stimes, axis = 1), axis = 1);
duration_min = np.min(duration);
dtmin = 1;

#stimes[1,0] = 60000;

stimes[:,-1] = stimes[:,0] + duration_min;
#stimes[:,0] = stimes[:,-1] - dtmin;

#stimes[:,-1] = stimes[:,0] + dtmin;
#stimes[:,0]-stimes[:,-1]


dt = 250;
sp = 10;
traj_hist = 120*60 / dt * sp;

times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)

ids = [np.where(wids == w)[0][0] for w in wids_large];
times[ids[1]] += 5000; # start has no images !



fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_life_persisters_median+2_cols.mov');
#fname = None;

def plot_number(row, col, ncols):
  return col + row * ncols+ 1;


#%%

reload(v);

fig = plt.figure(1, figsize = (5,3), dpi = 300, facecolor = 'w'); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.015, hspace = 0.075, wspace = 0.075)

def stage_stamp_position(xlim, ylim):
  return [xlim[1] + 90, ylim[0] + 30];

stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];

plots = [];
rows = 9; cols = 14 + 4;
col = 0; row = 0;
for wi,w in enumerate(wids):
  print col,row
  ax = plt.subplot(rows, cols, plot_number(row, col, cols));
  #col += 1;
  #if col % 14 == 0:
  #  row += 1;
  #  col = 0;
  
  row += 1;
  if row % rows == 0:
    row = 0;
    col += 1;  
  
  tcm = plt.cm.gray_r;
  #pcol = 'black'
  #if w in dwell:
  #  tcm = plt.cm.Reds;
  #  pcol = 'red';
  #elif w in roam:
  #  tcm = plt.cm.Blues;
  #  pcol = 'blue';
  pcol = plt.cm.coolwarm(1.0 * wi / len(wids));
  pcol = 'black';
  pl = 0.25;
  if w == wids_large[0]:
    pcol = 'red';
    pl = 0.5
  if w == wids_large[1]:
    pcol = 'blue';
    pl = 0.5;
  
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[wi,0],   
           image = False,
           
           plate_color = pcol,
           plate_linewidth = pl,

           trajectory = True,
           trajectory_length=traj_hist, trajectory_delta= dt/sp, trajectory_size=0.25, 
           trajectory_cmap= tcm,
           stage_stamp_font_size=2.8,
           stage_stamp_colors = stage_colors,
           
           stage_indicator = False,
           stage_indicator_label_font_size=10,
           
           stage_stamp = True,
           stage_stamp_position = stage_stamp_position,   
           
           time_stamp = False,
           time_stamp_font_size=12,
           time_stamp_offset= 34200,           
           
           border = [[15,15],[15,15]]);       
  plots.append(p);



#%%

from copy import copy


plots_all = copy(plots);
times_all = np.zeros((times.shape[0]+2, times.shape[1]), dtype = int);
times_all[:-2,:] = times;


for i,w in enumerate(wids_large):
  ax = plt.subplot2grid((8,cols), (4 * i, 14), colspan=4, rowspan = 4);
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[ids[i],0]+00000,   
           image = True,
           
           plate_color = np.array(['red', 'blue'])[i],
           plate_linewidth = 0.5,

           trajectory = True,
           trajectory_length=traj_hist, trajectory_delta= dt/sp, trajectory_size=1, 
           trajectory_cmap= np.array([plt.cm.Reds, plt.cm.Blues])[i],
           stage_stamp_font_size=8,
           
           stage_indicator = True,
           stage_indicator_label_font_size=5,
           stage_indicator_label_offset = 70,
           stage_indicator_size = 15,
           stage_indicator_colors = stage_colors,
           
           stage_stamp = False,
           stage_stamp_position = stage_stamp_position,  
           stage_stamp_colors = stage_colors,
           
           time_stamp = False,
           time_stamp_font_size=6,
           time_stamp_offset= 34200,           
           
           border = [[30,10],[120,10]]);       
  plots_all.append(p);
  times_all[-2+i,:] = times[ids[i]];



#%%

#times_short = times_all[:, :300];

def time_text(time, index):
  return 'N2 %s' % v.time_str(time);

a = v.WormAnimation(figure = fig, plots = plots_all, times = times_all +00000,
                    time_stamp = True,
                    time_stamp_text = time_text,
                    time_stamp_font_size = 5.5,
                    save = fname, fps = 30, dpi = 300);

fig.show()



#%%
a.animate()
 



#%%

















#%% All worms + 2 


import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);



#%% 

wids = np.array(range(125), dtype = int);


#%% persitence based ordering

order_mode = np.array(['index', 'median'])[1];

wids = np.array(np.genfromtxt('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Data/persistance_%s_order.csv' % order_mode), dtype = int);
wids = wids - 1;


#%%
order_mode = 'roam_mean'

wids = np.arange(125);
#wids[90] = 124;  # wid = 90 is corrupted data !

# sort via average roaming fraction
roaming = exp.load(strain = 'n2', wid = wids, dtype = 'roam');


roaming_mean = np.array([np.nanmean(r) for r in roaming]);

rorder = np.argsort(roaming_mean);

wids = np.array(wids)[rorder];


#%%

dwell = np.array([55, 72, 49, 51, 5, 69, 45, 73, 84, 95]) - 1;
roam  = np.array([23, 79, 123, 87, 48, 96, 26, 39, 9]) - 1;

wids_large = np.array([39, 51], dtype = int) - 1;
ids = [np.where(wids == w)[0][0] for w in wids_large];

stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
stimes[ids[1],0] += 5000; # start has no images !

sdurations = np.diff(stimes, axis = 1);
sdurations_max = np.max(sdurations, axis = 0);

nworms = len(wids);

aligned = 'raw';


delta_t = 2 * 60 * 3; #  sec * sample rate
fps = 30;
subsample = 2 *6;
dt = delta_t / subsample;
traj_hist = 30 # history of traj in minutes
traj_length = traj_hist * 3 * 60 / dt;
print 'trajectory history is %f min, sampled every %f secs = %d points' % ((1.0 * traj_length * dt / 3.0 / 60), 1.0 * dt /3, traj_length) ;
print 'frame time difference: %f sec' % (delta_t / 3.0)


if aligned == 'aligned':
  nstages = np.array(np.ceil(1.0 * sdurations_max / delta_t), dtype = int);
  ntot = np.sum(nstages);
  istages = np.hstack([0, np.cumsum(nstages)]);
  
  times = np.zeros((len(wids), ntot), dtype = int);
  for wi in range(len(wids)):
    for s in range(5):
      tdat = np.arange(stimes[wi][s], stimes[wi][s+1], delta_t);
      k = len(tdat);
      times[wi,istages[s]:(istages[s]+k)] = tdat;
      times[wi,(istages[s]+k):istages[s+1]] = tdat[-1];
else: # raw time
  duration = np.sum(np.diff(stimes, axis = 1), axis = 1);
  duration_min = np.min(duration);
  duration_max = np.max(duration);
  duration_max = 53*60*60*3;
  #stimes[:,-1] = stimes[:,0] + duration_min;
  ntimes = np.arange(0, duration_max, delta_t).shape[0];
  times = np.zeros((nworms, ntimes), dtype = int);
  for i in range(nworms):
    tms = np.arange(stimes[i,0], stimes[i,-1], delta_t)[:ntimes];
    times[i, :len(tms)] = tms;
    times[i, len(tms):] = tms[-1];
    
  #times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)    


rcorder = np.array(['rows', 'cols'])[0];

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_life_%s+2_%s_%s.mov' % (order_mode,rcorder, aligned));
#fname = None;

def plot_number(row, col, ncols):
  return col + row * ncols+ 1;


#%%

reload(v);

fig = plt.figure(1, figsize = (5,3), dpi = 300, facecolor = 'w'); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.015, hspace = 0.075, wspace = 0.075)

def stage_stamp_position(xlim, ylim):
  return [xlim[1] + 90, ylim[1] - 30];

stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];

plots = [];
rows = 9; cols = 14 + 4;
col = 0; row = 0;



for wi,w in enumerate(wids):
  print col,row
  ax = plt.subplot(rows, cols, plot_number(row, col, cols));
  
  if rcorder == 'rows':
    col += 1;
    if col % 14 == 0:
      row += 1;
      col = 0;
  else:
    row += 1;
    if row % rows == 0:
      row = 0;
      col += 1;  
  
  tcm = plt.cm.gray_r;
  #pcol = 'black'
  #if w in dwell:
  #  tcm = plt.cm.Reds;
  #  pcol = 'red';
  #elif w in roam:
  #  tcm = plt.cm.Blues;
  #  pcol = 'blue';
  pcol = plt.cm.coolwarm(1.0 * wi / len(wids));
  pcol = 'black';
  pl = 0.25;
  if w == wids_large[0]:
    pcol = 'red';
    pl = 0.5
  if w == wids_large[1]:
    pcol = 'blue';
    pl = 0.5;
    
  xy = exp.load(strain = 'n2', wid = w, dtype = 'xy');
  xy[:times[wi,0],:] = np.nan;
  
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[wi,0], xy = xy,
           image = False,
           
           plate_color = pcol,
           plate_linewidth = pl,

           trajectory = True,
           trajectory_length=traj_length, trajectory_delta= dt, trajectory_size=0.25, 
           trajectory_cmap= tcm,
           stage_stamp_font_size=2.8,
           stage_stamp_colors = stage_colors,
           
           stage_indicator = False,
           stage_indicator_label_font_size=10,
           
           stage_stamp = True,
           stage_stamp_position = stage_stamp_position,   
           
           time_stamp = False,
           time_stamp_font_size=12,
           time_stamp_offset= 34200,           
           
           border = [[15,15],[15,15]]);       
  plots.append(p);



#%%

from copy import copy


plots_all = copy(plots);
times_all = np.zeros((times.shape[0]+2, times.shape[1]), dtype = int);
times_all[:-2,:] = times;

def stage_indicator_position(time, time_max, time_min, xlim, ylim, time_end = None):
  sizefac= 0.9;
  plot_range = xlim[1] - xlim[0];
  off = xlim[0] + (1 - sizefac) * plot_range / 2.0;
  x = sizefac * (time - time_min) / (time_max - time_min) * plot_range + off;
  y = -40 + ylim[1];
  if time_end is None:
    return [x,y];
  else:
    h = 10;
    w = sizefac * (time_end - time) / (time_max - time_min) * plot_range;
    return x,y,w,h;



for i,w in enumerate(wids_large):
  ax = plt.subplot2grid((8,cols), (4 * i, 14), colspan=4, rowspan = 4);
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[ids[i],0]+00000,   
           image = True,
           
           plate_color = np.array(['red', 'blue'])[i],
           plate_linewidth = 0.5,

           trajectory = True,
           trajectory_length=traj_length, trajectory_delta= dt, trajectory_size=1, 
           trajectory_cmap= np.array([plt.cm.Reds, plt.cm.Blues])[i],
           stage_stamp_font_size=8,
           
           stage_indicator = True,
           stage_indicator_label_font_size=5,
           stage_indicator_label_offset = 90,
           stage_indicator_size = 15,
           stage_indicator_colors = stage_colors,
           stage_indicator_position = stage_indicator_position,

           
           stage_stamp = False,
           stage_stamp_position = stage_stamp_position,  
           stage_stamp_colors = stage_colors,
           
           time_stamp = False,
           time_stamp_font_size=6,
           time_stamp_offset= 34200,           
           
           border = [[30,10],[10,100]]);       
  plots_all.append(p);
  times_all[-2+i,:] = times[ids[i]];



#%%

#times_short = times_all[:, :300];

if aligned == 'aligned':
  nstages_acc = np.cumsum(nstages)
  stage_names = ['L1', 'L2', 'L3', 'L4', 'A'];
  def time_text(time, index):
    s = (nstages_acc <= index).sum();
    return 'N2 %s %s' % (stage_names[s], v.time_str((index - istages[s]) * dt));
else:
  def time_text(time, index):
    return 'N2 %s' % v.time_str(time);

a = v.WormAnimation(figure = fig, plots = plots_all, times = times_all[:,:],
                    time_stamp = True,
                    time_stamp_text = time_text,
                    time_stamp_font_size = 5.5,
                    save = fname, fps = 30, dpi = 300);

fig.show()



#%%
a.animate()
 





#%%





#%% All worms + 1


import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);


#%% persitence based ordering

order_mode = np.array(['index', 'median'])[1];

wids = np.array(np.genfromtxt('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Data/persistance_%s_order.csv' % order_mode), dtype = int);
wids = wids - 1;



#%%

dwell = np.array([55, 72, 49, 51, 5, 69, 45, 73, 84, 95]) - 1;
roam  = np.array([23, 79, 123, 87, 48, 96, 26, 39, 9]) - 1;

wids_large = np.array([39, 51, 36], dtype = int) - 1;
ids = [np.where(wids == w)[0][0] for w in wids_large];

stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
stimes[ids[1],0] += 5000; # start has no images !

sdurations = np.diff(stimes, axis = 1);
sdurations_max = np.max(sdurations, axis = 0);

nworms = len(wids);

aligned = 'aligned';


delta_t = 2 * 60 * 3; #  sec * sample rate
fps = 30;
subsample = 2 *6;
dt = delta_t / subsample;
traj_hist = 30 # history of traj in minutes
traj_length = traj_hist * 3 * 60 / dt;
print 'trajectory history is %f min, sampled every %f secs = %d points' % ((1.0 * traj_length * dt / 3.0 / 60), 1.0 * dt /3, traj_length) ;
print 'frame time difference: %f sec' % (delta_t / 3.0)


if aligned == 'aligned':
  nstages = np.array(np.ceil(1.0 * sdurations_max / delta_t), dtype = int);
  ntot = np.sum(nstages);
  istages = np.hstack([0, np.cumsum(nstages)]);
  
  times = np.zeros((len(wids), ntot), dtype = int);
  for wi in range(len(wids)):
    for s in range(5):
      tdat = np.arange(stimes[wi][s], stimes[wi][s+1], delta_t);
      k = len(tdat);
      times[wi,istages[s]:(istages[s]+k)] = tdat;
      times[wi,(istages[s]+k):istages[s+1]] = tdat[-1];
else: # raw time
  duration = np.sum(np.diff(stimes, axis = 1), axis = 1);
  duration_min = np.min(duration);
  duration_max = np.max(duration);
  duration_max = 53*60*60*3;
  #stimes[:,-1] = stimes[:,0] + duration_min;
  ntimes = np.arange(0, duration_max, delta_t).shape[0];
  times = np.zeros((nworms, ntimes), dtype = int);
  for i in range(nworms):
    tms = np.arange(stimes[i,0], stimes[i,-1], delta_t)[:ntimes];
    times[i, :len(tms)] = tms;
    times[i, len(tms):] = tms[-1];
    
  #times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)    


rcorder = np.array(['rows', 'cols'])[0];

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_life_%s+1_%s_%s.mov' % (order_mode,rcorder, aligned));
#fname = None;

def plot_number(row, col, ncols):
  return col + row * ncols+ 1;


#%%

reload(v);

fig = plt.figure(1, figsize = (5,3), dpi = 300, facecolor = 'w'); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.015, hspace = 0.075, wspace = 0.075)

def stage_stamp_position(xlim, ylim):
  return [xlim[1] + 90, ylim[1] - 30];

stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];

plots = [];
rows = 9; cols = 14 + 4;
col = 0; row = 0;



for wi,w in enumerate(wids):
  print col,row
  ax = plt.subplot(rows, cols, plot_number(row, col, cols));
  
  if rcorder == 'rows':
    col += 1;
    if col % 14 == 0:
      row += 1;
      col = 0;
  else:
    row += 1;
    if row % rows == 0:
      row = 0;
      col += 1;  
  
  tcm = plt.cm.gray_r;
  #pcol = 'black'
  #if w in dwell:
  #  tcm = plt.cm.Reds;
  #  pcol = 'red';
  #elif w in roam:
  #  tcm = plt.cm.Blues;
  #  pcol = 'blue';
  pcol = plt.cm.coolwarm(1.0 * wi / len(wids));
  pcol = 'black';
  pl = 0.25;
  if w == wids_large[0]:
    pcol = 'red';
    pl = 0.5
  if w == wids_large[1]:
    pcol = 'blue';
    pl = 0.5;
  if w == wids_large[2]:
    pcol = 'gray';
    pcol = (0.8, 0.8, 0.8);
    pl = 0.7;
    
  xy = exp.load(strain = 'n2', wid = w, dtype = 'xy');
  xy[:times[wi,0],:] = np.nan;
  
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[wi,0], xy = xy,
           image = False,
           
           plate_color = pcol,
           plate_linewidth = pl,

           trajectory = True,
           trajectory_length=traj_length, trajectory_delta= dt, trajectory_size=0.25, 
           trajectory_cmap= tcm,
           stage_stamp_font_size=2.8,
           stage_stamp_colors = stage_colors,
           
           stage_indicator = False,
           stage_indicator_label_font_size=10,
           
           stage_stamp = True,
           stage_stamp_position = stage_stamp_position,   
           
           time_stamp = False,
           time_stamp_font_size=12,
           time_stamp_offset= 34200,           
           
           border = [[15,15],[15,15]]);       
  plots.append(p);



#%%

from copy import copy


plots_all = copy(plots);
times_all = np.zeros((times.shape[0]+2, times.shape[1]), dtype = int);
times_all[:-2,:] = times;

def stage_indicator_position(time, time_max, time_min, xlim, ylim, time_end = None):
  sizefac= 0.9;
  plot_range = xlim[1] - xlim[0];
  off = xlim[0] + (1 - sizefac) * plot_range / 2.0;
  x = sizefac * (time - time_min) / (time_max - time_min) * plot_range + off;
  y = -40 + ylim[1];
  if time_end is None:
    return [x,y];
  else:
    h = 10;
    w = sizefac * (time_end - time) / (time_max - time_min) * plot_range;
    return x,y,w,h;

wids_large = [wids_large[-1]];
ids = [np.where(wids == w)[0][0] for w in wids_large];

for i,w in enumerate(wids_large):
  ax = plt.subplot2grid((8,cols), (4 * i + 2, 14), colspan=4, rowspan = 4);
  
  xy = exp.load(strain = 'n2', wid = w, dtype = 'xy');
  xy[:times[ids[i],0],:] = np.nan;  
  
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[ids[i],0], xy = xy,
           image = True,
           
           plate_color = np.array(['gray', 'red', 'blue'])[i],
           plate_linewidth = 0.5,

           trajectory = True,
           trajectory_length=traj_length, trajectory_delta= dt, trajectory_size=1, 
           trajectory_cmap= np.array([plt.cm.gray_r, plt.cm.Reds, plt.cm.Blues])[i],
           stage_stamp_font_size=8,
           
           stage_indicator = True,
           stage_indicator_label_font_size=5,
           stage_indicator_label_offset = 90,
           stage_indicator_size = 15,
           stage_indicator_colors = stage_colors,
           stage_indicator_position = stage_indicator_position,

           
           stage_stamp = False,
           stage_stamp_position = stage_stamp_position,  
           stage_stamp_colors = stage_colors,
           
           time_stamp = False,
           time_stamp_font_size=6,
           time_stamp_offset= 34200,           
           
           border = [[30,10],[10,100]]);       
  plots_all.append(p);
  times_all[-2+i,:] = times[ids[i]];



#%%

#times_short = times_all[:, :300];

if aligned == 'aligned':
  nstages_acc = np.cumsum(nstages)
  stage_names = ['L1', 'L2', 'L3', 'L4', 'A'];
  def time_text(time, index):
    s = (nstages_acc <= index).sum();
    return 'N2 %s %s' % (stage_names[s], v.time_str((index - istages[s]) * delta_t));
else:
  def time_text(time, index):
    return 'N2 %s' % v.time_str(time);

a = v.WormAnimation(figure = fig, plots = plots_all, times = times_all[:,:],
                    time_stamp = True,
                    time_stamp_text = time_text,
                    time_stamp_font_size = 5.5,
                    save = fname, fps = 30, dpi = 300);

fig.show()



#%%
a.animate()





#%%




#%% 100 worms + 1


import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);


#%% persitence based ordering

order_mode = np.array(['index', 'median'])[1];

wids = np.array(np.genfromtxt('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Data/persistance_%s_order.csv' % order_mode), dtype = int);
wids = wids - 1;

ids = np.array(np.linspace(0,len(wids)-1,100), dtype = int);

wids = wids[ids];




#%%

dwell = np.array([55, 72, 49, 51, 5, 69, 45, 73, 84, 95]) - 1;
roam  = np.array([23, 79, 123, 87, 48, 96, 26, 39, 9]) - 1;

wids_large = np.array([39, 51, 36], dtype = int) - 1;
ids = [np.where(wids == w)[0][0] for w in wids_large];

stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
stimes[ids[1],0] += 5000; # start has no images !

sdurations = np.diff(stimes, axis = 1);
sdurations_max = np.max(sdurations, axis = 0);

nworms = len(wids);

aligned = 'raw';


delta_t = 2 * 60 * 3; #  sec * sample rate
fps = 30;
subsample = 2 *6;
dt = delta_t / subsample;
traj_hist = 30 # history of traj in minutes
traj_length = traj_hist * 3 * 60 / dt;
print 'trajectory history is %f min, sampled every %f secs = %d points' % ((1.0 * traj_length * dt / 3.0 / 60), 1.0 * dt /3, traj_length) ;
print 'frame time difference: %f sec' % (delta_t / 3.0)


if aligned == 'aligned':
  nstages = np.array(np.ceil(1.0 * sdurations_max / delta_t), dtype = int);
  ntot = np.sum(nstages);
  istages = np.hstack([0, np.cumsum(nstages)]);
  
  times = np.zeros((len(wids), ntot), dtype = int);
  for wi in range(len(wids)):
    for s in range(5):
      tdat = np.arange(stimes[wi][s], stimes[wi][s+1], delta_t);
      k = len(tdat);
      times[wi,istages[s]:(istages[s]+k)] = tdat;
      times[wi,(istages[s]+k):istages[s+1]] = tdat[-1];
else: # raw time
  duration = np.sum(np.diff(stimes, axis = 1), axis = 1);
  duration_min = np.min(duration);
  duration_max = np.max(duration);
  duration_max = 53*60*60*3;
  #stimes[:,-1] = stimes[:,0] + duration_min;
  ntimes = np.arange(0, duration_max, delta_t).shape[0];
  times = np.zeros((nworms, ntimes), dtype = int);
  for i in range(nworms):
    tms = np.arange(stimes[i,0], stimes[i,-1], delta_t)[:ntimes];
    times[i, :len(tms)] = tms;
    times[i, len(tms):] = tms[-1];
    
  #times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)    


rcorder = np.array(['rows', 'cols'])[0];

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_life_100_%s+1_%s_%s.mov' % (order_mode,rcorder, aligned));
#fname = None;

def plot_number(row, col, ncols):
  return col + row * ncols+ 1;


#%%

reload(v);

fig = plt.figure(1, figsize = (5,3), dpi = 300, facecolor = 'w'); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.015, hspace = 0.075, wspace = 0.075)

def stage_stamp_position(xlim, ylim):
  return [xlim[1] + 90, ylim[1] - 30];

stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];

plots = [];
rows = 10; cols = 10 + 4;
col = 0; row = 0;



for wi,w in enumerate(wids):
  print col,row
  ax = plt.subplot(rows, cols, plot_number(row, col, cols));
  
  if rcorder == 'rows':
    col += 1;
    if col % 10 == 0:
      row += 1;
      col = 0;
  else:
    row += 1;
    if row % rows == 0:
      row = 0;
      col += 1;  
  
  tcm = plt.cm.gray_r;
  #pcol = 'black'
  #if w in dwell:
  #  tcm = plt.cm.Reds;
  #  pcol = 'red';
  #elif w in roam:
  #  tcm = plt.cm.Blues;
  #  pcol = 'blue';
  pcol = plt.cm.coolwarm(1.0 * wi / len(wids));
  pcol = 'black';
  pl = 0.25;
  if w == wids_large[0]:
    pcol = 'red';
    pl = 0.5
  if w == wids_large[1]:
    pcol = 'blue';
    pl = 0.5;
  if w == wids_large[2]:
    pcol = 'gray';
    pcol = (0.8, 0.8, 0.8);
    pl = 0.7;
    
  xy = exp.load(strain = 'n2', wid = w, dtype = 'xy');
  xy[:times[wi,0],:] = np.nan;
  
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[wi,0], xy = xy,
           image = False,
           
           plate_color = pcol,
           plate_linewidth = pl,

           trajectory = True,
           trajectory_length=traj_length, trajectory_delta= dt, trajectory_size=0.25, 
           trajectory_cmap= tcm,
           stage_stamp_font_size=2.8,
           stage_stamp_colors = stage_colors,
           
           stage_indicator = False,
           stage_indicator_label_font_size=10,
           
           stage_stamp = True,
           stage_stamp_position = stage_stamp_position,   
           
           time_stamp = False,
           time_stamp_font_size=12,
           time_stamp_offset= 34200,           
           
           border = [[15,15],[15,15]]);       
  plots.append(p);



#%%

from copy import copy


plots_all = copy(plots);
times_all = np.zeros((times.shape[0]+2, times.shape[1]), dtype = int);
times_all[:-2,:] = times;

def stage_indicator_position(time, time_max, time_min, xlim, ylim, time_end = None):
  sizefac= 0.9;
  plot_range = xlim[1] - xlim[0];
  off = xlim[0] + (1 - sizefac) * plot_range / 2.0;
  x = sizefac * (time - time_min) / (time_max - time_min) * plot_range + off;
  y = -40 + ylim[1];
  if time_end is None:
    return [x,y];
  else:
    h = 10;
    w = sizefac * (time_end - time) / (time_max - time_min) * plot_range;
    return x,y,w,h;

wids_large = [35];
ids = [np.where(wids == w)[0][0] for w in wids_large];

for i,w in enumerate(wids_large):
  ax = plt.subplot2grid((4,cols), (4 * i + 1, 10), colspan=4, rowspan = 2);
  p = v.WormPlot(ax, strain = 'n2', wid = w, time = times[ids[i],0]+00000,   
           image = True,
           
           plate_color = np.array(['gray', 'red', 'blue'])[i],
           plate_linewidth = 0.5,

           trajectory = True,
           trajectory_length=traj_length, trajectory_delta= dt, trajectory_size=1, 
           trajectory_cmap= np.array([plt.cm.gray_r, plt.cm.Reds, plt.cm.Blues])[i],
           stage_stamp_font_size=8,
           
           stage_indicator = True,
           stage_indicator_label_font_size=5,
           stage_indicator_label_offset = 90,
           stage_indicator_size = 15,
           stage_indicator_colors = stage_colors,
           stage_indicator_position = stage_indicator_position,

           
           stage_stamp = False,
           stage_stamp_position = stage_stamp_position,  
           stage_stamp_colors = stage_colors,
           
           time_stamp = False,
           time_stamp_font_size=6,
           time_stamp_offset= 34200,           
           
           border = [[30,10],[10,100]]);       
  plots_all.append(p);
  times_all[-2+i,:] = times[ids[i]];



#%%

#times_short = times_all[:, :300];

if aligned == 'aligned':
  nstages_acc = np.cumsum(nstages)
  stage_names = ['L1', 'L2', 'L3', 'L4', 'A'];
  def time_text(time, index):
    s = (nstages_acc <= index).sum();
    return 'N2 %s %s' % (stage_names[s], v.time_str((index - istages[s]) * delta_t));
else:
  def time_text(time, index):
    return 'N2 %s' % v.time_str(time);

a = v.WormAnimation(figure = fig, plots = plots_all, times = times_all[:,:],
                    time_stamp = True,
                    time_stamp_text = time_text,
                    time_stamp_font_size = 5.5,
                    save = fname, fps = 30, dpi = 300);

fig.show()



#%%
a.animate()









































