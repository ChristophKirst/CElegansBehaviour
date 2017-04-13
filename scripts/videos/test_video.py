# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 01:28:27 2017

@author: ckirst
"""

#%% Initialize

import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);


#%% Make a Movie
reload(v);

plt.figure(3, figsize = (2,2), dpi = 400, facecolor = 'w'); plt.clf();
plt.xticks([]); plt.yticks([]);
plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)

wid = 38;
wid = 50;
wid = 56;

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'life_wid=%d.mov' % (wid,));


stimes = exp.load_stage_times(wid = wid);
t0 = stimes[0];
t1 = stimes[-2];

#wid = 50;
#t0 = 262000;

#t0 = 552061;
#t0 = 34200;

dt = 250;
sp = 10;

hist = 120*60 / dt * sp;



#%%
reload(v)

#times = np.arange(t0, t1, dt)[:10];
times = np.arange(t0, t1, dt);
v.animate_frames(strain = 'n2', wid = wid, 
                 times = times, pause = None, 
                 history_delta = dt/sp, history = hist, 
                 features = [], feature_indicator= None,
                 time_data = None, time_size = 0.5, time_cmap = plt.cm.Reds,
                 border = 20, font_size=6, add_below = 100,
                 save = fname, fps = 10, dpi = 400,
                 time_stamp = True, stage_stamp=True)



#%% Worm Pair

reload(v)
wids = [38,50];
stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
duration = np.sum(np.diff(stimes, axis = 1), axis = 1);
duration_min = np.min(duration);
dtmin = 1;

stimes[1,0] = 60000;
#stimes[1,0] = 262000;

stimes[:,-1] = stimes[:,0] + duration_min;
#stimes[:,0] = stimes[:,-1] - dtmin;

#stimes[:,-1] = stimes[:,0] + dtmin;
#stimes[:,0]-stimes[:,-1]

dt = 250;
sp = 10;

hist = 120*60 / dt * sp;

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'life_wid=%r.mov' % (wids,));

times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)


#%%

reload(v)

plt.figure(1, figsize = (3.5,2.5), dpi = 360, facecolor = 'w'); plt.clf();
plt.xticks([]); plt.yticks([]);
plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)

ww = v.animate_worms(strain = 'n2', wid = wids, 
                 times = times, pause = 0.01, 
                 history_delta = dt/sp, history = hist, 
                 time_data = None, time_size = 0.5, time_cmap = plt.cm.Reds,
                 border = 60, vmin = 84, vmax = 92,
                 save = fname, fps = 10, dpi = 360,
                 time_stamp = True, font_size = 8, font_size_title = 12)



#%% Make Movie of 100 worms


reload(v)
wids = range(110);
wids[90] = 111;

# sort via average roaming fraction
roaming = exp.load(strain = 'n2', wid = wids, dtype = 'roam');


roaming_mean = np.array([np.nanmean(r) for r in roaming]);

rorder = np.argsort(roaming_mean);

wids = np.array(wids)[rorder];

#%%


stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];
duration = np.sum(np.diff(stimes, axis = 1), axis = 1);
duration_min = np.min(duration);
dtmin = 1;

stimes[50,0] = 60000;
#stimes[1,0] = 262000;

stimes[:,-1] = stimes[:,0] + duration_min;
#stimes[:,-1] = stimes[:,0] + dtmin;
#stimes[:,0]-stimes[:,-1]

dt = 300;
sp = 10;

hist = 120*60 / dt * sp;

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worms_life.mov');

times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,0], stimes[:,-1])], dtype = int)

#tt0 = 1500;
#times = times[:,tt0:(tt0+1)];

# algin to LXX
stage = 4;

stimes= exp.load_stage_times(strain = 'n2', wid = wids)[:,:-1];

duration = stimes[:,-1] - stimes[:,stage];
duration_min = np.min(duration);

stimes[:,-1] = stimes[:,stage] + duration_min;
#stimes[:,-1] = stimes[:,stage] + 10 * dt;

times = np.array([np.arange(ts,te,dt) for ts,te in zip(stimes[:,stage], stimes[:,-1])], dtype = int)


fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'worms_life_stage=%d.mov' % stage);


#%%
reload(v)
fig = plt.figure(1, figsize = (3.8,3.5), dpi = 300, facecolor = 'w'); plt.clf();
plt.xticks([]); plt.yticks([]);
plt.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0, hspace = 0.04, wspace = 0.04)


v.animate_worms(strain = 'n2', wid = wids, worm_image=False,
                 times = times, pause = 0.01, 
                 history_delta = dt/sp, history = hist, 
                 time_data = None, time_size = 0.2, time_cmap = plt.cm.Reds,
                 border = 10, vmin = 84, vmax = 92,
                 save = fname, fps = 10, dpi = 300, stage_indicator=False, stage_below=5,
                 line_width = 0.2, stamp_off = 70, extra_title = ('L%d' % (stage)),
                 time_stamp = True, font_size = 2, font_size_title = 4.5)
                 
                 
                 
#%% Make movie of roaming dwelling

wid = 38;

v.worm_feature_gui(strain = 'n2', wid = wid)

#%%


wid = 56;


dr = exp.load(strain = 'n2', wid = wid, dtype = 'roam');
plt.figure(10); plt.clf();
plt.plot(dr)


#%%

stimes = exp.load_stage_times(wid = wid);

#t0 = 34200+15000;
t0 = 70000 - 1000;
t0 = 180000
t0 = 263000-2000;
t0 = 365791
t0 = 524545
t0 = 526000
t0 = 520000
t0 = 541600;
t0 = 475500;
t0 = 514500;
t0 = 536500;
t0 = 491200;
t0 = 572500;
t0 = 416600;
t0 = 478000;
t0 = 496000; ###
t0 = 540000;
t0 = 500850;
t0 = 501050;
t0 = 501450;

sp = 3; #[Hz]
dt = 1 * sp
dt = 1;
hist = 5*60 / dt * sp;

t1 = int(t0 + 4 *60 * sp);

fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'roaming_dwelling_wid=%r.mov' % (wid));
#fname = os.path.join('/home/ckirst/Science/Projects/CElegans/Analysis/WormBehaviour/Movies', 'roaming_dwelling_wid=%r.avi' % (wid));


plt.figure(1, figsize = (2,2), dpi = 400, facecolor = 'w'); plt.clf();
plt.figure(1, figsize = (3,3), dpi = 400, facecolor = 'w'); plt.clf();

plt.xticks([]); plt.yticks([]);
plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)

#make the colormap 
from matplotlib.colors import LinearSegmentedColormap
cdict = {'red':   ((0.0,  0.5, 0.5),
                   (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.5, 0.5),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.5, 0.5),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0))}
cm = LinearSegmentedColormap('GrayRed', cdict)

timedata = exp.load(strain = 'n2', wid = wid, dtype = 'roam');
timedata[:t0] = np.nan;


#reload(v)
v.animate_frames(strain = 'n2', wid = wid, 
                 times = range(t0, t1, dt), pause = None, 
                 history_delta = dt, history = hist, 
                 features = [], feature_indicator= None,
                 time_data = timedata, time_size = 0.5, time_cmap = cm, tref = stimes[0],
                 border = 20, font_size = 7,
                 save = fname, fps = 10, dpi = 400, legend = [('gray', 'dwelling'),('r', 'roaming')],
                 time_stamp = True)





#%%

import numpy as np
import os
import analysis.experiment as exp
import analysis.plot as fplt
import matplotlib.pyplot as plt  

import analysis.video as vd
reload(vd)

from analysis.features import moving_average
  
dat_v = exp.load(wid =0, strain = 'n2', dtype = 'speed')    
dat_v = moving_average(dat_v, bin_size=30);

dat_r = exp.load(wid =0, strain = 'n2', dtype = 'rotation')    
dat_r = moving_average(dat_r, bin_size=30);

w = vd.worm_feature_gui(strain = 'n2', wid = 38, history = 500, features = [('speed', dat_v), ('rotation', dat_r), 'roam'], levels = (60,90), background = None, feature_indicator="roam");


#%%


#os.system("mencoder 'mf://movie_wid=%d_*.png' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o worm_%d.mpg" % (wid,wid))
#import glob
#files = np.sort(np.unique(glob.glob('movie_wid=%d_*.png' % wid)))
#for f in files:
#  os.remove(f)






plt.figure(15); plt.clf();

roam = exp.load(wid = 0, dtype ='roam');
xy = exp.load(wid = 0, dtype = 'xy');

t0 = 550000;
t1 = t0 + 10000;
sh = 0;


plt.plot(xy[t0:t1:3, 0], xy[t0:t1:3,1], c = 'gray')
plt.scatter(xy[t0:t1:3, 0], xy[t0:t1:3,1], c = roam[t0+sh:t1+sh:3], s = 20)













xy = exp.load(wid = 0, dtype = 'xy')
t0 = 591963;
t0 = 600000;
t0 = 600702;
t0 = 600499;

t0 = 517694; # shift right -> 

print('>')
print(xy[t0])
print(xy[t0+1])

t0 = 517742; # shfit down v

print('v')
print(xy[t0])
print(xy[t0+1])





w1 = exp.load_img(t = t0);
w2 = exp.load_img(t = t0+1);
w3 = exp.load_img(t = t0+2);

plt.figure(1); plt.clf();

plt.imshow(w1)
plt.imshow(w2)



plt.imshow(w1[0:, :-1])
plt.imshow(w2[:, 1:])
plt.imshow(w3)


t0 = 517742;
w0 = np.array(exp.load_img(t = t0), dtype = float);
for i,d in enumerate([[0,0], [-1,0], [1,0], [0,-1], [0,1], [-1,-1], [-1,1], [1,-1], [1,1]]):
  plt.subplot(3,3,i+1);
  ww = w0.copy();
  ww = np.roll(ww, d, axis = [0,1]);
  plt.imshow(w0-ww)
  plt.title('%r, %d' % (d, np.sum(np.square(w0-ww))));







strain = 'n2'; wid = 0;

trans = np.load(os.path.join(exp.data_directory, 'transitions_times.npy'))
trans = trans[wid];
t0 = trans[0];

reload(vd)


speed = exp.load(strain = strain, wid = wid, dtype = 'speed')
plt.figure(6); plt.clf();
plt.plot(speed)
plt.ylim(0,20)

t0= 350600
vid = [vd.wormimage(t = t0+ti) for ti in range(0,1000,5)]
vid = np.array(vid)

fplt.pg.image(vid)  
plt.figure(7); plt.clf();
plt.imshow(vid[0], cmap = plt.cm.gray)

t0= 350600
img = vd.wormimage(t = t0, strain = strain, wid = wid, border = 75)
plt.figure(7); plt.clf(); 
plt.subplot(1,2,1)
plt.imshow(img, cmap = plt.cm.gray)
sh = img.shape;
plt.gca().add_artist(plt.Circle((sh[0]/2,sh[1]/2),sh[0]/2-75, color = 'black',fill = False, linewidth = 1))

reload(vd)
plt.figure(10); plt.clf();
vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+500, 5),
                  history= 50, history_delta=5)


file_format = 'movie_%d.png';
plt.figure(11);
t0 = 350600;
vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+5000, 1),
                  history= 150, history_delta=1, save = file_format);





#os.system("convert _tmp*.png animation.mng")

# cleanup
#for fname in files:
#  os.remove(fname)



#  import matplotlib
#  matplotlib.use("Agg")
#  import matplotlib.animation as manimation
#
#  FFMpegWriter = manimation.writers['ffmpeg']
#  metadata = dict(title='Worm %d t0 = %d' % (wid, t0), artist='Christoph Kirst', comment='CElegans Behaviour Project')
#  writer = FFMpegWriter(fps=15, metadata=metadata)
#
#  fig = plt.figure(11);
#  writer.saving(fig, "writer_test.mp4", 100);
#  vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+500, 5),
#                    history= 50, history_delta=5, writer= writer)




#%%

