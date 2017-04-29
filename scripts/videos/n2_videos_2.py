


#%%


import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);



#%% Animate two worms
wids = np.array([39, 51], dtype = int) - 1;
wids = [w for w in wids];

aligned = 'aligned';

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worm_%r_life_%s.mov' % (wids, aligned));



#%%

stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
#stimes[ids[1],0] += 5000; # start has no images !
stimes[-1,0] += 4900;

sdurations = np.diff(stimes, axis = 1);
sdurations_max = np.max(sdurations, axis = 0);

nworms = len(wids);


stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];


delta_t = 2 * 60 * 3; #  sec * sample rate
fps = 30;
subsample = 12;
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
  #duration_max = 53*60*60*3;
  #stimes[:,-1] = stimes[:,0] + duration_min;
  ntimes = np.arange(0, duration_max, delta_t).shape[0];
  times = np.zeros((nworms, ntimes), dtype = int);
  for i in range(nworms):
    tms = np.arange(stimes[i,0], stimes[i,-1], delta_t)[:ntimes];
    times[i, :len(tms)] = tms;
    times[i, len(tms):] = tms[-1];
    
  #times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)    



#%%

reload(v);
fig = plt.figure(2, figsize = (3.5,2), dpi = 360, facecolor = 'w'); plt.clf();
plt.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.035, hspace = 0.075, wspace = 0.075)



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

nworms = len(wids);
plts = [];
for i,w in enumerate(wids):
  if i== 1:
    pcol = 'blue';
    cm = plt.cm.Blues;
  else:
    pcol = 'red';
    cm = plt.cm.Reds;
    
  if i == 0:
    ts = True;
  else:
    ts = False;
  
  ax = plt.subplot(1,nworms, i+1);
  w = v.WormPlot(ax, strain = 'n2', wid = w, time = stimes[i,0], # time = tt,#
               image_filter=image_filter, image_sigma = None, image_vmin = 84, image_vmax = 92,
               
               plate_color = pcol, plate_linewidth = 0.5,   
               
               time_stamp_offset= stimes[i,0],
              
               trajectory_length = traj_length, trajectory_delta = dt, trajectory_size = 1, 
               trajectory_cmap= cm,
               
               stage_stamp_font_size = 6, stage_stamp_colors= stage_colors,
               stage_indicator_label_font_size = 5.5, stage_indicator_colors=stage_colors,
               stage_indicator_label_offset = 55,
               
               time_stamp = False,
               time_stamp_font_size=6, 
  
               border = [[5,5],[5,100]]);
  plts.append(w);
              
               

#%%

if aligned == 'aligned':
  nstages_acc = np.cumsum(nstages)
  stage_names = ['L1', 'L2', 'L3', 'L4', 'A'];
  def time_text(time, index):
    s = (nstages_acc <= index).sum();
    return 'N2 %s %s' % (stage_names[s], v.time_str((index - istages[s]) * delta_t));
else:
  def time_text(time, index):
    return 'N2 %s' % v.time_str(time);

   
a = v.WormAnimation(figure = fig, plots = plts, times = times,
                    fps = fps, dpi = 360, save = fname, time_stamp = True, time_stamp_font_size=7, time_stamp_text = time_text);

fig.show()

#%%

a.animate()
